#include <cmath>
#include <vector>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include "cnpy.h"

__global__ void fsm_kernel(float* T, float* S, int* sgnv, int* sgnt, int sgni, int sgnj,
                           int x_offset, int z_offset, int xd, int zd,
                           int nxx, int nzz, float dx, float dz, float dx2i, float dz2i)
{
    int element = blockIdx.x * blockDim.x + threadIdx.x;

    int i = z_offset + zd * element;
    int j = x_offset + xd * element;

    if ((i > 0) && (i < nzz - 1) && (j > 0) && (j < nxx - 1)) {
        int i1 = i - sgnv[sgni];
        int j1 = j - sgnv[sgnj];

        float tv = T[i - sgnt[sgni] + j * nzz];
        float te = T[i + (j - sgnt[sgnj]) * nzz];
        float tev = T[(i - sgnt[sgni]) + (j - sgnt[sgnj]) * nzz];

        float t1d1 = tv + dz * fminf(S[i1 + max(j - 1, 1) * nzz], S[i1 + min(j, nxx - 1) * nzz]);
        float t1d2 = te + dx * fminf(S[max(i - 1, 1) + j1 * nzz], S[min(i, nzz - 1) + j1 * nzz]);

        float t1D = fminf(t1d1, t1d2);
        float t1 = 1e6f, t2 = 1e6f, t3 = 1e6f;

        float Sref = S[i1 + j1 * nzz];

        if ((tv <= te + dx * Sref) && (te <= tv + dz * Sref) &&
            (te - tev >= 0.0f) && (tv - tev >= 0.0f)) {
            float ta = tev + te - tv;
            float tb = tev - te + tv;

            float disc = 4.0f * Sref * Sref * (dz2i + dx2i) - dz2i * dx2i * (ta - tb) * (ta - tb);
            if (disc > 0.0f) {
                t1 = ((tb * dz2i + ta * dx2i) + sqrtf(disc)) / (dz2i + dx2i);
            }
        }
        else if ((te - tev <= Sref * dz * dz / sqrtf(dx * dx + dz * dz)) && (te - tev > 0.0f)) {
            t2 = te + dx * sqrtf(Sref * Sref - ((te - tev) / dz) * ((te - tev) / dz));
        }
        else if ((tv - tev <= Sref * dx * dx / sqrtf(dx * dx + dz * dz)) && (tv - tev > 0.0f)) {
            t3 = tv + dz * sqrtf(Sref * Sref - ((tv - tev) / dx) * ((tv - tev) / dx));
        }

        float t2D = fminf(t1, fminf(t2, t3));

        T[i + j * nzz] = fminf(T[i + j * nzz], fminf(t1D, t2D));
    }
}

extern "C" {
float* fast_sweeping_method(float* Vp, float sx, float sz, float dx, float dz, int nx, int nz)
{
    int nb = 2;
    int nxx = nx + 2 * nb;
    int nzz = nz + 2 * nb;
    int matsize = nxx * nzz;

    float* T = new float[matsize]();
    float* S = new float[matsize]();
    float* eikonal = new float[nx * nz]();

    // Set slowness field S from velocity
    for (int i = 0; i < nz; i++) {
        for (int j = 0; j < nx; j++) {
            S[(i + nb) + (j + nb) * nzz] = 1.0f / Vp[i + j * nz];
        }
    }

    // Expand boundaries (mirror)
    for (int i = 0; i < nb; i++) {
        for (int j = nb; j < nxx - nb; j++) {
            S[i + j * nzz] = S[nb + j * nzz];
            S[(nzz - i - 1) + j * nzz] = S[(nzz - nb - 1) + j * nzz];
        }
    }
    for (int i = 0; i < nzz; i++) {
        for (int j = 0; j < nb; j++) {
            S[i + j * nzz] = S[i + nb * nzz];
            S[i + (nxx - j - 1) * nzz] = S[i + (nxx - nb - 1) * nzz];
        }
    }

    int sIdx = (int)(sx / dx) + nb;
    int sIdz = (int)(sz / dz) + nb;

    for (int index = 0; index < matsize; index++) T[index] = 1e6f;

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int xi = sIdx + (j - 1);
            int zi = sIdz + (i - 1);

            T[zi + xi * nzz] = S[zi + xi * nzz] * sqrtf(powf((xi - nb) * dx - sx, 2.0f) +
                                                        powf((zi - nb) * dz - sz, 2.0f));
        }
    }

    int nSweeps = 4;
    int meshDim = 2;
    int nThreads = 32;
    float dz2i = 1.0f / (dz * dz);
    float dx2i = 1.0f / (dx * dx);
    int min_level = std::min(nxx, nzz);
    int max_level = std::max(nxx, nzz);
    int total_levels = (nxx - 1) + (nzz - 1);

    std::vector<std::vector<int>> sgnv = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
    std::vector<std::vector<int>> sgnt = {{1, 1}, {-1, 1}, {1, -1}, {-1, -1}};

    int* h_sgnv = new int[nSweeps * meshDim]();
    int* h_sgnt = new int[nSweeps * meshDim]();

    for (int index = 0; index < nSweeps * meshDim; index++) {
        int j = index / nSweeps;
        int i = index % nSweeps;

        h_sgnv[i + j * nSweeps] = sgnv[i][j];
        h_sgnt[i + j * nSweeps] = sgnt[i][j];
    }

    std::vector<std::vector<int>>().swap(sgnv);
    std::vector<std::vector<int>>().swap(sgnt);

    // GPU memory allocation
    float* d_T = nullptr;
    float* d_S = nullptr;
    int* d_sgnv = nullptr;
    int* d_sgnt = nullptr;

    cudaMalloc((void**)&d_T, matsize * sizeof(float));
    cudaMalloc((void**)&d_S, matsize * sizeof(float));
    cudaMalloc((void**)&d_sgnv, nSweeps * meshDim * sizeof(int));
    cudaMalloc((void**)&d_sgnt, nSweeps * meshDim * sizeof(int));

    cudaMemcpy(d_T, T, matsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_S, S, matsize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sgnv, h_sgnv, nSweeps * meshDim * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sgnt, h_sgnt, nSweeps * meshDim * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_sgnv;
    delete[] h_sgnt;

    // Fast sweeping loop
    for (int sweep = 0; sweep < nSweeps; sweep++) {
        int zd = (sweep == 2 || sweep == 3) ? -1 : 1;
        int xd = (sweep == 0 || sweep == 2) ? -1 : 1;
        int sgni = sweep;
        int sgnj = sweep + nSweeps;

        for (int level = 0; level < total_levels; level++) {
            int z_offset = (sweep == 0) ? ((level < nxx) ? 0 : level - nxx + 1) :
                            (sweep == 1) ? ((level < nzz) ? nzz - level - 1 : 0) :
                            (sweep == 2) ? ((level < nzz) ? level : nzz - 1) :
                                           ((level < nxx) ? nzz - 1 : nzz - 1 - (level - nxx + 1));

            int x_offset = (sweep == 0) ? ((level < nxx) ? level : nxx - 1) :
                            (sweep == 1) ? ((level < nzz) ? 0 : level - nzz + 1) :
                            (sweep == 2) ? ((level < nzz) ? nxx - 1 : nxx - 1 - (level - nzz + 1)) :
                                           ((level < nxx) ? nxx - level - 1 : 0);

            int n_elements = (level < min_level) ? level + 1 :
                             (level >= max_level) ? total_levels - level :
                             total_levels - min_level - max_level + level;

            int nBlocks = (n_elements + nThreads - 1) / nThreads;

            fsm_kernel<<<nBlocks, nThreads>>>(d_T, d_S, d_sgnv, d_sgnt, sgni, sgnj,
                                              x_offset, z_offset, xd, zd,
                                              nxx, nzz, dx, dz, dx2i, dz2i);

            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(T, d_T, matsize * sizeof(float), cudaMemcpyDeviceToHost);

    #pragma omp parallel for
    for (int index = 0; index < nx * nz; index++) {
        int x = index / nz;
        int z = index % nz;
        eikonal[z + x * nz] = T[(z + nb) + (x + nb) * nzz];
    }

    // 🧹 Free device memory
    cudaFree(d_T);
    cudaFree(d_S);
    cudaFree(d_sgnv);
    cudaFree(d_sgnt);

    // 🧹 Free host memory
    delete[] T;
    delete[] S;

    return eikonal;
}

void free_eikonal(float* ptr) {
    delete[] ptr;
}
}


extern "C" {
    void reset_device() {
        cudaError_t err = cudaDeviceReset();
        // Optionally, you can check for errors:
        if (err != cudaSuccess) {
            // Handle error as needed
        }
    }
}


extern "C" {

    // Kernel that performs on-the-fly migration for one output pixel (as shown previously)
    __global__ void _migrate_constant_velocity(const float* data,
                                                const float* cdp,
                                                const float* offsets,
                                                float v,
                                                float dt, float dx, float dz,
                                                int nsmp, int ntraces,
                                                int nx, int nz,
                                                float* R)
    {
        // Each thread computes one output pixel (ix, iz)
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iz = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix >= nx || iz >= nz) return;
    
        float x = ix * dx;
        float z = iz * dz;
        float sum = 0.0f;
        
        // Loop over traces
        for (int j = 0; j < ntraces-1; j++) {
            float cdp_val = cdp[j];    // effective CDP for trace j
            float h = offsets[j] * 0.5f; // half offset for trace j
            // float doffset = fabsf(offsets[j+1]-offsets[j]);
            float source = cdp_val - h;
            float receiver = cdp_val + h;
            
            // Compute distances:
            float dxs = x - source;
            float dxg = x - receiver;
            float rs = sqrtf(dxs*dxs + z*z);
            float rr = sqrtf(dxg*dxg + z*z);
            float eps = 1e-7f;
            if (rs < eps) rs = eps;
            if (rr < eps) rr = eps;
            
            // Compute two-way travel time:
            float t_val = (rs + rr) / v;
            int it = (int) floorf(t_val / dt);
            if (it < 0 || it >= nsmp) continue;
            
            // Get seismic amplitude: data is stored as row-major (nsmp, ntraces)
            float amp = data[j * nsmp + it];
            
            // Compute weight (geometric spreading correction)
            float sqrt_rs_rr = sqrtf(rs/rr);
            float sqrt_rr_rs = 1.0f/sqrt_rs_rr;
            float weight = ((z/rs)*sqrt_rs_rr + (z/rr)*sqrt_rr_rs) / v;
            weight *= 0.3989422804f;  // 1/sqrt(2*pi)
            
            sum += amp * weight;
        }
        // Write the accumulated sum to the output migrated image R (assume row-major, shape (nx, nz))
        R[ix * nz + iz] = sum;
    }
    
    // Host-callable wrapper function for migration
    // This function allocates memory, sets up kernel launch parameters, and calls the kernel.
    void migrate_constant_velocity(const float* data, const float* cdp, const float* offsets,
                                   float v, float dt, float dx, float dz,
                                   int nsmp, int ntraces, int nx, int nz,
                                   float* R)
    {
        // Define block and grid dimensions.
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
        
        // Launch the kernel.
        _migrate_constant_velocity<<<grid, block>>>(data, cdp, offsets, v, dt, dx, dz,
                                                     nsmp, ntraces, nx, nz, R);
        cudaDeviceSynchronize();
    }

}


extern "C" {

    // Kernel that performs on-the-fly migration for one output pixel (as shown previously)
    __global__ void _migrate_variable_velocity(const float* data,
                                                const float* cdp,
                                                const float* offsets,
                                                float v,
                                                float dt, float dx, float dz,
                                                int nsmp, int ntraces,
                                                int nx, int nz,
                                                float* R)
    {
        // Each thread computes one output pixel (ix, iz)
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iz = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix >= nx || iz >= nz) return;
    
        float x = ix * dx;
        float z = iz * dz;
        float sum = 0.0f;
        
        // Loop over traces
        for (int j = 0; j < ntraces-1; j++) {
            float cdp_val = cdp[j];    // effective CDP for trace j
            float h = offsets[j] * 0.5f; // half offset for trace j
            // float doffset = fabsf(offsets[j+1]-offsets[j]);
            float source = cdp_val - h;
            float receiver = cdp_val + h;
            
            // Compute distances:
            float dxs = x - source;
            float dxg = x - receiver;
            float rs = sqrtf(dxs*dxs + z*z);
            float rr = sqrtf(dxg*dxg + z*z);
            float eps = 1e-7f;
            if (rs < eps) rs = eps;
            if (rr < eps) rr = eps;
            
            // Compute two-way travel time:
            float t_val = (rs + rr) / v;
            int it = (int) floorf(t_val / dt);
            if (it < 0 || it >= nsmp) continue;
            
            // Get seismic amplitude: data is stored as row-major (nsmp, ntraces)
            float amp = data[j * nsmp + it];
            
            // Compute weight (geometric spreading correction)
            float sqrt_rs_rr = sqrtf(rs/rr);
            float sqrt_rr_rs = 1.0f/sqrt_rs_rr;
            float weight = ((z/rs)*sqrt_rs_rr + (z/rr)*sqrt_rr_rs) / v;
            weight *= 0.3989422804f;  // 1/sqrt(2*pi)
            
            sum += amp * weight;
        }
        // Write the accumulated sum to the output migrated image R (assume row-major, shape (nx, nz))
        R[ix * nz + iz] = sum;
    }
    
    // Host-callable wrapper function for migration
    // This function allocates memory, sets up kernel launch parameters, and calls the kernel.
    void migrate_variable_velocity(const float* data, const float* cdp, const float* offsets,
                                   float v, float dt, float dx, float dz,
                                   int nsmp, int ntraces, int nx, int nz,
                                   float* R)
    {
        // Define block and grid dimensions.
        dim3 block(16, 16);
        dim3 grid((nx + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
        
        // Launch the kernel.
        _migrate_constant_velocity<<<grid, block>>>(data, cdp, offsets, v, dt, dx, dz,
                                                     nsmp, ntraces, nx, nz, R);
        cudaDeviceSynchronize();
    }

}