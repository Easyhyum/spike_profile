// gtstamp.cu — write the GPU %globaltimer (ns, same clock as freqmon) into out[idx] from the current stream.
// Launched right before each matmul burst so freqmon samples can be placed on the burst time axis.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void stamp_k(long long* __restrict__ out, int idx) {
    if (threadIdx.x == 0) {
        long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g));
        out[idx] = g;
    }
}
void stamp(torch::Tensor out, int64_t idx) {
    stamp_k<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((long long*)out.data_ptr(), (int)idx);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "stamp launch");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("stamp", &stamp); }
