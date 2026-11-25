// freqmon_excl.cu — freqmon (1-SM in-kernel clock monitor, %globaltimer + clock64) that EXCLUSIVELY
// owns its SM. Identical sampling to prof_freq/.../freqmon.cu, plus:
//   * 48 KB static shared memory: the RTX 5060 Ti has 100 KB smem/SM, so a GEMM CTA that itself
//     needs >52 KB (ours: 96 KB) can never be co-scheduled on the monitor's SM -> the GEMM really
//     has 36-1 = 35 SMs, which is what the wave count must be based on.
//   * records its own %smid in count[1] so the workload can prove its CTAs never ran there.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void freq_monitor_x(long long* __restrict__ buf, int* __restrict__ count,
                               const volatile int* __restrict__ stop,
                               int max_samples, long long interval_ns) {
    extern __shared__ volatile char pad[];          // occupancy blocker, size chosen at launch (never optimized away)
    if (threadIdx.x != 0) return;
    pad[0] = 1;
    unsigned smid; asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    count[1] = (int)smid;
    long long gt_prev;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt_prev));
    int i = 0;
    while (i < max_samples) {
        long long gt;
        long long target = gt_prev + interval_ns;
        do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(gt));
             if (gt < target) __nanosleep(500);
        } while (gt < target && !(*stop));
        if (*stop) break;
        long long c = (long long)clock64();
        buf[2*i] = gt; buf[2*i+1] = c;
        gt_prev = gt; i++;
    }
    count[0] = i + (int)pad[0] - 1;
}
void mon_start(torch::Tensor buf, torch::Tensor count, torch::Tensor stop,
               int64_t max_samples, int64_t interval_ns, int64_t smem_bytes) {
    if (smem_bytes > 48 * 1024)
        cudaFuncSetAttribute(freq_monitor_x, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    freq_monitor_x<<<1, 32, (size_t)smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        (long long*)buf.data_ptr(), (int*)count.data_ptr(),
        (const volatile int*)stop.data_ptr(), (int)max_samples, (long long)interval_ns);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess, "monitor launch");
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mon_start", &mon_start, py::arg("buf"), py::arg("count"), py::arg("stop"), py::arg("max_samples"),
          py::arg("interval_ns"), py::arg("smem_bytes") = 48 * 1024);
}
