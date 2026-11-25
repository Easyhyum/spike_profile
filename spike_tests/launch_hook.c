#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stddef.h>
typedef int CUresult; typedef void* CUfunction; typedef void* CUstream;
typedef struct { unsigned gx, gy, gz, bx, by, bz, smem; CUstream s; void* attrs; unsigned nattrs; } CUlaunchConfig;
typedef CUresult (*lk_t)(CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
typedef CUresult (*lkex_t)(const CUlaunchConfig*, CUfunction, void**, void**);
typedef CUresult (*fgn_t)(const char**, CUfunction);
typedef CUresult (*occ_t)(int*, CUfunction, int, size_t);
typedef void* (*dlsym_t)(void*, const char*);
static dlsym_t real_dlsym; static void* cuda_h; static lk_t real_lk, real_lk_ptsz; static lkex_t real_lkex, real_lkex_ptsz; static fgn_t fgn; static occ_t occ; static int tried;
static void get_dlsym(void) { if (!real_dlsym) real_dlsym = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5"); if (!real_dlsym) real_dlsym = (dlsym_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34"); }
static int depth;
static void report(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem) {
    depth++; if (depth > 3) { if (depth == 4) fprintf(stderr, "[hook] RECURSION depth=%d real_lk=%p\n", depth, (void*)real_lk); if (depth > 6) abort(); }
    if (!tried && cuda_h) { tried = 1; fgn = (fgn_t)real_dlsym(cuda_h, "cuFuncGetName"); occ = (occ_t)real_dlsym(cuda_h, "cuOccupancyMaxActiveBlocksPerMultiprocessor"); }
    const char* n = "?"; int nb = -1;
    if (fgn) fgn(&n, f);
    if (occ) occ(&nb, f, bx*by*bz, smem);
    if (!strstr(n, "stamp_k") && !strstr(n, "freqmon") && !getenv("HOOK_QUIET"))
        fprintf(stderr, "[launch] d=%d %s grid=%u,%u,%u (%u CTAs) block=%u,%u,%u (%u thr) dynsmem=%u maxCTAs/SM=%d\n", depth, n, gx, gy, gz, gx*gy*gz, bx, by, bz, bx*by*bz, smem, nb);
}
#define LK(name, real) static CUresult name(CUfunction f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned smem, CUstream s, void** kp, void** ex) { report(f, gx, gy, gz, bx, by, bz, smem); CUresult r = real(f, gx, gy, gz, bx, by, bz, smem, s, kp, ex); depth--; return r; }
#define LKEX(name, real) static CUresult name(const CUlaunchConfig* c, CUfunction f, void** kp, void** ex) { report(f, c->gx, c->gy, c->gz, c->bx, c->by, c->bz, c->smem); CUresult r = real(c, f, kp, ex); depth--; return r; }
LK(my_lk, real_lk) LK(my_lk_ptsz, real_lk_ptsz) LKEX(my_lkex, real_lkex) LKEX(my_lkex_ptsz, real_lkex_ptsz)
void* dlsym(void* h, const char* n) {
    get_dlsym(); void* p = real_dlsym(h, n);
    if (!p || !n) return p;
    if (!strncmp(n, "cuLaunchKernel", 14)) { Dl_info di; const char* lib = (dladdr(p, &di) && di.dli_fname) ? di.dli_fname : "?";
        if (!getenv("HOOK_QUIET")) fprintf(stderr, "[hook] dlsym(%s) h=%p p=%p from %s\n", n, h, p, lib);
        /* hook only the innermost driver library (the one that does not itself forward through dlsym) */
        if (!strstr(lib, "/drivers/")) return p; }
    if (!strcmp(n, "cuLaunchKernel"))         { cuda_h = h; real_lk = (lk_t)p; return (void*)my_lk; }
    if (!strcmp(n, "cuLaunchKernel_ptsz"))    { cuda_h = h; real_lk_ptsz = (lk_t)p; return (void*)my_lk_ptsz; }
    if (!strcmp(n, "cuLaunchKernelEx"))       { cuda_h = h; real_lkex = (lkex_t)p; return (void*)my_lkex; }
    if (!strcmp(n, "cuLaunchKernelEx_ptsz"))  { cuda_h = h; real_lkex_ptsz = (lkex_t)p; return (void*)my_lkex_ptsz; }
    return p;
}
