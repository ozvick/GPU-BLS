# gpu-bls

GPU-accelerated Box Least Squares (BLS) period search for TESS light curves, written in CUDA C++.

**Status:** Week 1 of 6 — CUDA fundamentals (`warmup/`).

## Layout
- `warmup/` — learning exercises: vector addition, SAXPY, reductions
- (coming) `cpu_ref/` — CPU reference BLS validated against astropy
- (coming) `kernels/` — the GPU BLS kernel and optimizations

## Week 1 results: vector addition (n = 2^24 floats)
GPU: NVIDIA H200 NVL · CUDA 13.0 · `nvcc -O3`

| | Time | vs CPU |
|---|---|---|
| CPU loop | 6.15 ms | 1x |
| GPU kernel only | 0.067 ms | ~92x faster |
| GPU incl. host<->device copies | 11.7 ms | ~1.9x slower |

The kernel reaches ~3.0 TB/s effective bandwidth (~60% of H200 peak), but
vector addition does one add per 12 bytes moved, so PCIe transfers dominate.
Takeaway for BLS: copy the light curve once and keep it on the GPU.

## Running on UT Dallas Juno
```bash
module load cuda/13.0
nvcc -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 warmup/vecadd.cu -o warmup/vecadd
cd warmup && sbatch run_vecadd.sh
```

Computations performed on the Juno cluster, HPC@UTD.
