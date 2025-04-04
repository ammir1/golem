#include <cmath>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <nppi.h>
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cerrno>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <npp.h>
// #include <nppi.h>
// #include <nppi_resize.h>  // Specifically required for nppiResize_32f_C1R with scaling

// Error check macro
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

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

    // Kernel that performs on-the-fly migration for one output pixel (as shown previously)
    __global__ void _migrate_constant_velocity(const float* __restrict__ data,
                                                const float* __restrict__ cdp,
                                                const float* __restrict__ offsets,
                                                float v,
                                                float dt, float dx, float dz,
                                                int nsmp, int ntraces,
                                                int nx, int nz,
                                                float* __restrict__ R)
    {
        // Each thread computes one output pixel (ix, iz)
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iz = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix >= nx || iz >= nz) return;
    
        float x = ix * dx;
        float z = iz * dz;
        float sum = 0.0f;
        
        // Loop over traces
        #pragma unroll 9
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
            float eps = 1e-10f;
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
        R[ix + iz * nx] = sum;
    }

    
    // Host-callable wrapper function for migration
    // This function allocates memory, sets up kernel launch parameters, and calls the kernel.
    void migrate_constant_velocity(const float* data, const float* cdp, const float* offsets,
                                   float v, float dt, float dx, float dz,
                                   int nsmp, int ntraces, int nx, int nz,
                                   float* R)
    {
        // Define block and grid dimensions.
        dim3 block(32, 8);
        dim3 grid((nx + block.x - 1) / block.x, (nz + block.y - 1) / block.y);
        // dim3 grid((ntraces + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
        
        // Launch the kernel.
        _migrate_constant_velocity<<<grid, block>>>(data, cdp, offsets, v, dt, dx, dz,
                                                     nsmp, ntraces, nx, nz, R);
        cudaDeviceSynchronize();
    }

}




//----------------------------------------------------------------------------
// Helper structure and functions for mapping a file into pinned memory.
// A simple structure to hold mapped file info.

extern "C" void init_cuda_with_mapped_host() {
    cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaSetDeviceFlags failed: " << cudaGetErrorString(err) << std::endl;
    }
}

// Function: Load binary file into heap-allocated buffer
float* load_binary_file(const char* filename, size_t* out_count)
{
    FILE* file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Get size in bytes and compute number of float elements
    fseek(file, 0, SEEK_END);
    size_t filesize = ftell(file);
    rewind(file);

    if (filesize % sizeof(float) != 0) {
        fprintf(stderr, "File size is not a multiple of float size\n");
        fclose(file);
        return NULL;
    }

    *out_count = filesize / sizeof(float);

    float* data = (float*)malloc(filesize);
    if (!data) {
        perror("malloc failed");
        fclose(file);
        return NULL;
    }

    size_t read_count = fread(data, sizeof(float), *out_count, file);
    fclose(file);

    if (read_count != *out_count) {
        fprintf(stderr, "fread failed: expected %zu floats, got %zu\n", *out_count, read_count);
        free(data);
        return NULL;
    }

    return data;
}

// Function: Copy data from host to device
float* copy_to_device(const float* host_data, size_t count)
{
    float* device_data = NULL;
    CHECK_CUDA(cudaMalloc((void**)&device_data, count * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_data, host_data, count * sizeof(float), cudaMemcpyHostToDevice));
    return device_data;
}


void resample_field_into_npp_cubic(float* d_base,
                                   float* d_dst,
                                   size_t offset_index,
                                   int nx_coarse, int nz_coarse,
                                   int nx_fine, int nz_fine)
{
    // Compute pointer to the selected field
    size_t field_size = nx_coarse * nz_coarse;
    float* d_src = d_base + offset_index * field_size;

    // Set up NPP image sizes
    NppiSize src_size = { nx_coarse, nz_coarse };
    NppiSize dst_size = { nx_fine, nz_fine };

    // Define source and destination ROIs
    NppiRect src_roi = { 0, 0, nx_coarse, nz_coarse };
    NppiRect dst_roi = { 0, 0, nx_fine, nz_fine };

    int src_step = nx_coarse * sizeof(float);
    int dst_step = nx_fine * sizeof(float);

    // Perform the resize operation
    NppStatus status = nppiResize_32f_C1R(
        d_src, src_step, src_size, src_roi,
        d_dst, dst_step, dst_size, dst_roi,
        NPPI_INTER_CUBIC
    );

    if (status != NPP_SUCCESS) {
        fprintf(stderr, "nppiResize_32f_C1R failed with code: %d\n", status);
    }
}
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Host-callable migration function.
// This version re-reads (via pointer arithmetic) a traveltime field from a memory-mapped file
// for each update, resamples it to fine dimensions using NPP, and calls the migration kernel.
// Parameters:
//  - cdp, offsets, v, dt, dx, dz, nsmp, ntraces: parameters for migration.
//  - nx_coarse, nz_coarse: dimensions of each coarse traveltime field in the file.
//  - nx_fine, nz_fine: desired fine output dimensions.
//  - R: output migrated image (device pointer).
//  - traveltime_filename: path to the binary file with appended traveltime fields.
//  - num_fields: number of traveltime fields in the file.

extern "C" {

    // Kernel that performs on-the-fly migration for one output pixel (as shown previously)
    __global__ void _migrate_variable_velocity(const float* __restrict__ data,
                                                const float* __restrict__ cdp,
                                                const float* __restrict__ offsets,
                                                const float* __restrict__ SourceTraveltime,
                                                const float* __restrict__ ReceiverTraveltime,
                                                float v,
                                                float dt, float dx, float dz,
                                                int nsmp, int ntraces, int init_traces,
                                                int nx, int nz,
                                                float* __restrict__ R)
    {
        // Each thread computes one output pixel (ix, iz)
        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iz = blockIdx.y * blockDim.y + threadIdx.y;
        if (ix >= nx || iz >= nz) return;
    
        float x = ix * dx;
        float z = iz * dz;
        float sum = 0.0f;
        
        // Loop over traces
        #pragma unroll 9
        for (int j = init_traces; j < init_traces+ntraces; j++) {
            
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
            float eps = 1e-10f;
            if (rs < eps) rs = eps;
            if (rr < eps) rr = eps;

            float s_traveltime = 1.0f / SourceTraveltime[ix + iz * nx];
            float r_traveltime = 1.0f / ReceiverTraveltime[ix + iz * nx];

            
            // Compute two-way travel time:
            float t_val = s_traveltime + r_traveltime;
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
        R[ix + iz * nx] = sum;
    }
}


extern "C" void migrate_variable_velocity(const float* data, const float* cdp, const float* offsets, 
                                            const float* eikonal_positions, const int* segments,
                                            float v, float dt, float dx, float dz,
                                            int nsmp, int ntraces,
                                            int nx_coarse, int nz_coarse,
                                            int nx_fine, int nz_fine,
                                            float* R,
                                            const char* traveltime_filename,
                                            const char* gradient_filename,
                                            int num_segments)
{


    // Assume each coarse traveltime field has the same size:

    size_t num_floats = 226*nx_coarse * nz_coarse * sizeof(float);

    // allocated fine traveltime
    float* d_SourceTravelTime_fine = NULL;
    float* d_ReceiverTravelTime_fine = NULL;

    // size_t fine_size_bytes = nx_fine * nz_fine * sizeof(float);
    // cudaMalloc((void**)&d_SourceTravelTime_fine, num_floats);
    // cudaMalloc((void**)&d_ReceiverTravelTime_fine, num_floats);

    float* h_SourceTravelTimeCoarse = load_binary_file(traveltime_filename, &num_floats);
    if (!h_SourceTravelTimeCoarse) {
        fprintf(stderr, "Failed to load file.\n");
        // return 1;
    }

    float* d_ReceiverTravelTime_fine = load_binary_file(traveltime_filename, &num_floats);

    if (!h_TravelTimeCoarse) {
        fprintf(stderr, "Failed to load file.\n");
        // return 1;
    }


    float* d_TravelTimeCoarse = copy_to_device(h_TravelTimeCoarse, num_floats);

    int old_src_index = -1;
    int old_rec_index = -1;
    size_t init_traces = 0;
    for (int field = 0; field < num_segments; field++) {
        
        int n_traces = segments[field];

        float sx = cdp[init_traces] - 0.5f * offsets[init_traces];
        float rx = cdp[init_traces] + 0.5f * offsets[init_traces];

        int src_index = static_cast<int>(sx/300.0f);
        int rec_index = static_cast<int>(rx/300.0f);

        float x_x0_src = eikonal_positions[src_index] - sx;
        float x_x0_rec = eikonal_positions[rec_index] - rx;

        // if (old_src_index != src_index)
        // {
        //     resample_field_into_npp_cubic(d_TravelTimeCoarse, d_SourceTravelTime_fine,  src_index, 
        //         nx_coarse, nz_coarse, nx_fine, nz_fine);
            
        //     old_src_index = src_index;
        // }
            
        // if (old_rec_index != rec_index)
        // {
        //     resample_field_into_npp_cubic(d_TravelTimeCoarse, d_ReceiverTravelTime_fine,  rec_index, 
        //         nx_coarse, nz_coarse, nx_fine, nz_fine);
            
        //     old_rec_index = rec_index;
        // }
            
        dim3 block(16, 16);
        dim3 grid((nx_fine + block.x - 1) / block.x, (nz_fine + block.y - 1) / block.y);
        // dim3 grid((ntraces + block.x - 1) / block.x, (nx + block.y - 1) / block.y);
        
        // Launch the kernel.
        _migrate_variable_velocity<<<grid, block>>>(data, cdp, offsets, d_SourceTravelTime_fine, 
                                                    d_ReceiverTravelTime_fine, v, dt, dx, dz,
                                                     nsmp, ntraces, init_traces, nx_fine, nz_fine, R);
        cudaDeviceSynchronize();
        init_traces += n_traces;
        
    }

    cudaFree(d_SourceTravelTime_fine);
    cudaFree(d_ReceiverTravelTime_fine);
    cudaFree(d_TravelTimeCoarse);
    free(h_TravelTimeCoarse);
}



