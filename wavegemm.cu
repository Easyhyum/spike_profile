// wavegemm.cu — fp32 GEMM C[MxN] = A[MxK] * B[KxN] (row-major) with an explicit, predictable grid for the wave sweep.
// Same blocking as the cuBLAS kernel on this GPU (cutlass simt sgemm 256x128): threadblock tile 256x128x8, 256 threads
// = 8 warps (4 along M x 2 along N, warp tile 64x64), thread tile 16x8, k-tile 16, FFMA from double-buffered shared memory with
// float4 global loads and LDS.128 operand reads. Dynamic smem is padded to 72 KB so that exactly ONE CTA fits per SM
// (checked with the occupancy API; freqmon's 96 KB SM cannot host one either), hence waves = (M/256)(N/128) / usable SMs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#define TM 256
#define TN 128
#define TK 16
#define SMEM_PAD_BYTES (72 * 1024)
#define STAGE_FLOATS (TK * TM + TK * TN)     // As[TK][TM] + Bs[TK][TN] = 6144 floats = 24 KB per stage (2 stages = 48 KB)

__global__ void __launch_bounds__(256, 1) wavegemm_k(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    extern __shared__ __align__(16) float dyn[];
    const int bm = blockIdx.y * TM, bn = blockIdx.x * TN;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 1, wn = warp & 1;             // 4 x 2 warps
    const int ty = lane >> 3, tx = lane & 7;             // 4 x 8 threads in a warp
    const int row0 = wm * 64 + ty * 16, col0 = wn * 64 + tx * 8;
    float acc[16][8];
    #pragma unroll
    for (int i = 0; i < 16; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
    // global -> register staging: A: thread tid loads row tid of the 256x8 tile (2 x float4); B: row tid/32, cols (tid%32)*4 (1 x float4)
    const float* Ag = A + (size_t)(bm + tid) * K;
    const float* Bg = B + (size_t)(tid >> 5) * N + bn + (lane << 2);
    float4 ra[TK / 4], rb[TK / 8];
    auto gload = [&](int k0) {
        #pragma unroll
        for (int i = 0; i < TK / 4; i++) ra[i] = *reinterpret_cast<const float4*>(Ag + k0 + 4 * i);
        #pragma unroll
        for (int j = 0; j < TK / 8; j++) rb[j] = *reinterpret_cast<const float4*>(Bg + (size_t)(k0 + 8 * j) * N);
    };
    auto sstore = [&](int buf) {
        float* As = dyn + buf * STAGE_FLOATS; float* Bs = As + TK * TM;
        #pragma unroll
        for (int i = 0; i < TK / 4; i++) { As[(4 * i) * TM + tid] = ra[i].x; As[(4 * i + 1) * TM + tid] = ra[i].y; As[(4 * i + 2) * TM + tid] = ra[i].z; As[(4 * i + 3) * TM + tid] = ra[i].w; }
        #pragma unroll
        for (int j = 0; j < TK / 8; j++) *reinterpret_cast<float4*>(Bs + ((tid >> 5) + 8 * j) * TN + (lane << 2)) = rb[j];
    };
    gload(0); sstore(0); __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < K; k0 += TK) {
        const bool more = (k0 + TK < K);
        if (more) gload(k0 + TK);                         // prefetch the next k-tile while this one computes
        const float* As = dyn + buf * STAGE_FLOATS; const float* Bs = As + TK * TM;
        #pragma unroll
        for (int kk = 0; kk < TK; kk++) {
            float a[16], b[8];
            const float4* ap = reinterpret_cast<const float4*>(As + kk * TM + row0);
            const float4* bp = reinterpret_cast<const float4*>(Bs + kk * TN + col0);
            #pragma unroll
            for (int i = 0; i < 4; i++) { float4 v = ap[i]; a[4 * i] = v.x; a[4 * i + 1] = v.y; a[4 * i + 2] = v.z; a[4 * i + 3] = v.w; }
            #pragma unroll
            for (int j = 0; j < 2; j++) { float4 v = bp[j]; b[4 * j] = v.x; b[4 * j + 1] = v.y; b[4 * j + 2] = v.z; b[4 * j + 3] = v.w; }
            #pragma unroll
            for (int i = 0; i < 16; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        if (more) sstore(buf ^ 1);
        __syncthreads(); buf ^= 1;
    }
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        float* cp = C + (size_t)(bm + row0 + i) * N + bn + col0;
        *reinterpret_cast<float4*>(cp) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(cp + 4) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

static bool attr_set = false;
static void ensure_attr() {
    if (!attr_set) { cudaFuncSetAttribute(wavegemm_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_PAD_BYTES); attr_set = true; }
}

void gemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda() && A.dtype() == torch::kFloat32, "fp32 CUDA tensors");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K && C.size(0) == M && C.size(1) == N, "shape");
    TORCH_CHECK(M % TM == 0 && N % TN == 0 && K % TK == 0, "M multiple of 256, N multiple of 128, K multiple of 8");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "contiguous");
    ensure_attr();
    dim3 grid(N / TN, M / TM), block(256);
    wavegemm_k<<<grid, block, SMEM_PAD_BYTES, at::cuda::getCurrentCUDAStream()>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

int max_blocks_per_sm() { ensure_attr(); int nb = 0; cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, wavegemm_k, 256, SMEM_PAD_BYTES); return nb; }
int sm_count() { int d = 0; cudaGetDevice(&d); int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d); return n; }
int smem_per_sm() { int d = 0; cudaGetDevice(&d); int n = 0; cudaDeviceGetAttribute(&n, cudaDevAttrMaxSharedMemoryPerMultiprocessor, d); return n; }
int tile_m() { return TM; }
int tile_n() { return TN; }
int regs_per_thread() { cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, wavegemm_k); return fa.numRegs; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "C = A @ B (fp32, 256x128 tile per CTA, 1 CTA/SM)");
    m.def("max_blocks_per_sm", &max_blocks_per_sm); m.def("sm_count", &sm_count); m.def("smem_per_sm", &smem_per_sm);
    m.def("tile_m", &tile_m); m.def("tile_n", &tile_n); m.def("regs_per_thread", &regs_per_thread);
}
