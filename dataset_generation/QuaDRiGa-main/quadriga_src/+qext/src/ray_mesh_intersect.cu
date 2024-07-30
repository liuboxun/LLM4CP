#include <iostream>
#include <chrono>
#include <cmath>
#include "qd_mesh_functions.h"
#include "ray_mesh_intersect.h"
using namespace std;

//Define an assert style handler function and wrapper macro to check for errors in runtime API code
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
        cout << "GPUassert: '" << cudaGetErrorString(code) << "' in " << file << ":" << line << endl;
}

// Block dimensions must be declared statically to correcly initiate shared memory
#define BLOCK_SIZE_RayMesh 256 // Block size for "RayMeshIntersect"
#define BLOCK_SIZE_RayRx 512   // Block size for "RayRxIntersect"
#define SERIAL_RAYS_RayRx 32   // Number of rays to be processed in serial in a Kernel

// KERNEL: Calculate "dest" = "dest" - "orig"
__global__ void DestMinusOrig(float *Ox, float *Oy, float *Oz, float *Dx, float *Dy, float *Dz)
{
    unsigned int i_ray = blockIdx.x * blockDim.x + threadIdx.x; // Ray index
    Dx[i_ray] = Dx[i_ray] - Ox[i_ray];
    Dy[i_ray] = Dy[i_ray] - Oy[i_ray];
    Dz[i_ray] = Dz[i_ray] - Oz[i_ray];
}

// KERNEL: Test if a ray hits a mesh element
__global__ void RayMeshIntersect(float *v1x, float *v1y, float *v1z, float *v2x, float *v2y, float *v2z,
                                 float *v3x, float *v3y, float *v3z, float *Ox, float *Oy, float *Oz,
                                 float *Dx, float *Dy, float *Dz, float *Wx, float *Wy, unsigned int *Wz, unsigned int *hit_cnt,
                                 float *Wout, unsigned int no_hit_W)
{
    // Input variables:
    //  v1x - v3z       Mesh coordinates, vectors of lenght [no_mesh]
    //  Ox, Oy, Oz      Ray origin, vectors of length [no_rays]
    //  Dx, Dy, Dz      Ray destinations, vectors of length [no_rays]
    //  no_hit_W        Number of interactions to be retunrned in the output

    // Output variables:
    //  Wx              Minimum normalized distance, values between 0 and 1, vector of lenght [no_ray_chnk * no_mesh_blk]
    //  Wy              Maximum normalized distance, values between 0 and 1, vector of lenght [no_ray_chnk * no_mesh_blk]
    //  Wz              Index of the first mesh element that was hit by the ray, vector of length [no_ray_chnk * no_mesh_blk]
    //  hit_cnt         Number of mesh indersections, vector of length [no_ray]
    //  Wout            List of normalized distances, values between 0 and 1, vector of lenght [no_ray * no_hit_W]

    // Get the current ray and mesh element
    unsigned int i_ray = blockIdx.x;
    unsigned int i_mesh = blockIdx.y * BLOCK_SIZE_RayMesh + threadIdx.x;

    // Shared variable to locally store results and ray data of one block
    __shared__ float Wshared[BLOCK_SIZE_RayMesh];
    __shared__ unsigned int Windex[BLOCK_SIZE_RayMesh];
    __shared__ unsigned int no_hit;
    __shared__ float k_Dx, k_Dy, k_Dz, k_Ox, k_Oy, k_Oz;

    // First thread initializs the local hit counter in shared memory
    if (threadIdx.x == 0)
        no_hit = 0;

    // Read one ray coordinate per warp and store it in shared memory
    if (threadIdx.x == 32)
        k_Dx = Dx[i_ray];
    if (threadIdx.x == 64)
        k_Dy = Dy[i_ray];
    if (threadIdx.x == 96)
        k_Dz = Dz[i_ray];
    if (threadIdx.x == 128)
        k_Ox = Ox[i_ray];
    if (threadIdx.x == 160)
        k_Oy = Oy[i_ray];
    if (threadIdx.x == 192)
        k_Oz = Oz[i_ray];

    // Synchronize so that all threads have access to the shared data
    __syncthreads();

    // Calculate Vector from V1 to O
    float Tx = k_Ox - v1x[i_mesh];
    float Ty = k_Oy - v1y[i_mesh];
    float Tz = k_Oz - v1z[i_mesh];

    // Read edges V1-V2 and and V1-V3 from global memory
    float k_v2x = v2x[i_mesh];
    float k_v2z = v2z[i_mesh];
    float k_v2y = v2y[i_mesh];
    float k_v3x = v3x[i_mesh];
    float k_v3y = v3y[i_mesh];
    float k_v3z = v3z[i_mesh];

    // Calculate 1st barycentric coordinate (gU)
    float PQ = k_v3z * k_Dy - k_v3y * k_Dz;
    float R0 = 3.74e-23;
    float DT = R0 + k_v2x * PQ;
    float U = Tx * PQ;

    PQ = k_v3x * k_Dz - k_v3z * k_Dx;
    DT = DT + k_v2y * PQ;
    U = U + Ty * PQ;
    PQ = k_v3y * k_Dx - k_v3x * k_Dy;
    DT = DT + k_v2z * PQ;
    U = U + Tz * PQ;

    DT = 1 / DT;
    U = U * DT;

    // Calculate and 2nd barycentric coordinate (gV)
    // Calculate and normalized line intersect position (gW)
    PQ = k_v2z * Ty - k_v2y * Tz;
    float V = PQ * k_Dx;
    float W = PQ * k_v3x;

    PQ = k_v2x * Tz - k_v2z * Tx;
    V = V + PQ * k_Dy;
    W = W + PQ * k_v3y;

    PQ = k_v2y * Tx - k_v2x * Ty;
    V = V + PQ * k_Dz;
    W = W + PQ * k_v3z;

    V = V * DT;
    W = W * DT;

    // The number of hits is very low - atomic functions should not cause many collisions
    if (U >= 0 && V >= 0 && (U + V) <= 1 && W > 0 && W <= 1) // Intersect condition
    {
        unsigned int ind = atomicInc(&no_hit, BLOCK_SIZE_RayMesh); // Local hit counter for current block
        Wshared[ind] = W;                                          // Store all hits of current block in shared memory
        Windex[ind] = i_mesh;                                      // Index of the hit
        ind = atomicInc(&hit_cnt[i_ray], UINT_MAX);                // Global hit counter for all rays
        if (ind < no_hit_W)                                        // Store first hits in global memory
            Wout[i_ray * no_hit_W + ind] = W;
    }

    // Synchronize to make sure all results are ready before writing the output to global memory
    __syncthreads();

    // Thread 0 of each block processes the partial result and writes it to global memory
    float Wtx = 1;        // Minimun
    float Wty = 0;        // Maximum
    unsigned int Wtz = 0; // Index
    if (threadIdx.x == 0 && no_hit > 0)
    {
        for (unsigned int i = 0; i < no_hit; i++)
        {
            Wtx = (Wshared[i] < Wtx) ? Wshared[i] : Wtx;
            Wty = (Wshared[i] > Wty) ? Wshared[i] : Wty;
            Wtz = (Wshared[i] == Wtx) ? Windex[i] : Wtz;
        }
        Wx[gridDim.x * blockIdx.y + i_ray] = Wtx;
        Wy[gridDim.x * blockIdx.y + i_ray] = Wty;
        Wz[gridDim.x * blockIdx.y + i_ray] = Wtz;
    }
}

// KERNEL: Calculate 3D FBS and LBS positions, accumulate per-block-results
__global__ void CalcIntersectPoints(unsigned int no_mesh_blk, float *Ox, float *Oy, float *Oz,
                                    float *Dx, float *Dy, float *Dz, float *Wx, float *Wy, unsigned int *Wz, unsigned int *iFBS)
{
    unsigned int i_ray = blockIdx.x * blockDim.x + threadIdx.x; // Ray index
    unsigned int n_ray = gridDim.x * 32;                        // Number of rays

    float Wtx; // Minimun
    float Wty; // Maximum
    float W_min = 1;
    float W_max = 0;
    unsigned int W_ind = 0;

    // Read first and last interaction point and number of interactions
    for (unsigned int i = 0; i < no_mesh_blk; i++)
    {
        Wty = Wy[i * n_ray + i_ray]; // Global memory read maximum
        if (Wty > 0)
        {
            Wtx = Wx[i * n_ray + i_ray]; // Global memory read minimum
            W_min = (Wtx < W_min) ? Wtx : W_min;
            W_max = (Wty > W_max) ? Wty : W_max;
            W_ind = (Wtx == W_min) ? Wz[i * n_ray + i_ray] + 1 : W_ind; // Global memory read index
        }
        Wy[i * n_ray + i_ray] = 0; // Reset for next chunk
    }

    // Calculate FBS and LBS - overwrite orig and dest in global memory
    float k_Ox = Ox[i_ray];
    float k_Dx = Dx[i_ray];
    Ox[i_ray] = k_Ox + W_min * k_Dx;

    float k_Oy = Oy[i_ray];
    float k_Dy = Dy[i_ray];
    Oy[i_ray] = k_Oy + W_min * k_Dy;

    float k_Oz = Oz[i_ray];
    float k_Dz = Dz[i_ray];
    Oz[i_ray] = k_Oz + W_min * k_Dz;

    Dx[i_ray] = k_Ox + W_max * k_Dx;
    Dy[i_ray] = k_Oy + W_max * k_Dy;
    Dz[i_ray] = k_Oz + W_max * k_Dz;

    iFBS[i_ray] = W_ind;
}


// FUNCTION: Calculate ray-mesh interset points
void ray_mesh_intersect_CUDA(Matrix orig, Matrix dest, Matrix mesh, Matrix fbs, Matrix lbs,
                             unsigned int *hit, unsigned int *iFBS, Matrix Wout, int verbose)
{
    // Input variables:
    //  orig            Ray origin, matrix of size [no_ray x 3]
    //  dest            Ray destinations, matrix of size [no_ray x 3]
    //  mesh            Mesh coordinates (v1 and e12, e13), matrix of size [no_mesh x 9]
    //  verbose         Enables or disbles progress report

    // Output variables:
    //  fbs             First interaction point of the ray with the mesh, matrix of size [no_ray x 3]
    //  lbs             Last interaction point of the ray with the mesh, matrix of size [no_ray x 3]
    //  hit             Number of mesh indersections, vector of length [no_ray]
    //  iFBS            Index of the first mesh element that was hit by the ray, vector of length [no_ray]
    //  Wout            List of normalized distances, values between 0 and 1, matrix of size  [no_hit_W x no_ray]

    // Check inputs
    if (orig.width != 3 || dest.width != 3 || fbs.width != 3 || lbs.width != 3)
    {
        cout << "Error: 'orig', 'dest', 'fbs' and 'lbs' must have 3 columns!" << endl;
        return;
    }
    if (dest.height != orig.height || fbs.height != orig.height || lbs.height != orig.height || Wout.width != orig.height)
    {
        cout << "Error: 'orig', 'dest', 'fbs', 'lbs' and 'Wout' must have the same number of rays!" << endl;
        return;
    }
    if (mesh.width != 9)
    {
        cout << "Error: 'mesh' must have 9 rows!" << endl;
        return;
    }

    // choose which GPU to run on
    gpuErrchk(cudaSetDevice(0));

    // Set scheduler to use less CPU  for busy wait loop
    gpuErrchk(cudaSetDeviceFlags(cudaDeviceBlockingSync));

    // Determine availabe memory on device
    size_t free;                                                      // Free device memory in byte
    size_t total;                                                     // Total device memory in byte
    gpuErrchk(cudaMemGetInfo(&free, &total));                         // Get device memory info
    float freeGB = floor(((float)free) / 1024 / 1024 / 102.4) / 10;   // Free device memory in GB
    float totalGB = floor(((float)total) / 1024 / 1024 / 102.4) / 10; // Total device memory in GB
    float device_mem = freeGB - 0.3;                                  // Used device memory in GB (max. 16 GB)
    device_mem = (device_mem > 15.9) ? 15.9 : device_mem;

    // Progress report
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    float vb_dots = 50;
    float m0 = 0;
    if (verbose == 1)
        cout << "LOS det. GPU [" << flush;
    else if (verbose == 2)
        cout << "Avail. memory:  " << freeGB << " GB of " << totalGB << " GB, using max. " << device_mem << " GB" << endl;

    unsigned int no_ray = orig.height;                                                // Number of rays, must be the same in orig and dest
    unsigned int no_ray_d = (no_ray / 32 + 1) * 32;                                   // Number of rays on device, multiple of 32 (warp-size)
    unsigned int no_hit_W = Wout.height;                                              // Number of interactions to be retunrned in the output
    unsigned int no_mesh = mesh.height;                                               // Mesh size
    unsigned int no_mesh_d = (no_mesh / BLOCK_SIZE_RayMesh + 1) * BLOCK_SIZE_RayMesh; // Mesh size on device
    unsigned int no_mesh_blk = no_mesh_d / BLOCK_SIZE_RayMesh;                        // Number of mesh-blocks

    device_mem = (device_mem - 0.1) * 256 * 1024 * 1024;                                    // Number of 32 bit words on device
    unsigned int mem_size = (unsigned int)device_mem;                                       // Number of 32 bit words as unsigned int
    mem_size = mem_size - 9 * no_mesh_d;                                                    // Subtract size of the mesh
    unsigned int no_ray_chnk = (mem_size / (8 + 3 * no_mesh_blk + no_hit_W) / 32 + 1) * 32; // Number of rays to be processed in parallel
    no_ray_chnk = (no_ray_chnk > no_ray_d) ? no_ray_d : no_ray_chnk;                        // Limit to no_ray_d if only one chunk
    unsigned int no_chnk = (unsigned int)ceil((float)no_ray_d / (float)no_ray_chnk);        // Number of chunks

    // Calculate the needed device memory
    mem_size = 9 * no_mesh_d + no_ray_chnk * (8 + 3 * no_mesh_blk + no_hit_W); // 32 Bit words
    device_mem = ceil(4 * (float)mem_size / 1024 / 1024);                      // MB

    if (verbose == 2)
    {
        cout << "Used memory:    " << device_mem << " MB" << endl;
        cout << "Mesh size:      " << no_mesh << " (" << no_mesh_d << ", " << no_mesh_blk << " blocks)" << endl;
        cout << "Number of rays: " << no_ray << " (" << no_ray_d << ")" << endl;
        cout << "Chunk size:     " << no_ray_chnk << " (" << no_chnk << " chunks)" << endl;
    }

    // Allocate device memory
    float *d_v1x, *d_v1y, *d_v1z, *d_v2x, *d_v2y, *d_v2z, *d_v3x, *d_v3y, *d_v3z; // Vertices
    float *d_Ox, *d_Oy, *d_Oz, *d_Dx, *d_Dy, *d_Dz;                               // Rays
    float *d_Wx, *d_Wy;                                                           // Internal storage for partial results
    unsigned int *d_Wz;                                                           // Internal storage for index
    unsigned int *d_hit_cnt;                                                      // Counter for the number of hits
    unsigned int *d_iFBS;                                                         // Index of first hit
    float *d_Wout;                                                                // Normalized intersection coordinates

    size_t size_vert_h = no_mesh * sizeof(float);   // Mesh size on host
    size_t size_vert_d = no_mesh_d * sizeof(float); // Mesh size on device
    size_t size_ray = no_ray_chnk * sizeof(float);  // Ray size
    size_t size_W = no_ray_chnk * no_mesh_blk * sizeof(float);
    size_t size_Wz = no_ray_chnk * no_mesh_blk * sizeof(unsigned int);
    size_t size_cnt = no_ray_chnk * sizeof(unsigned int);
    size_t size_Wout = no_hit_W * no_ray_chnk * sizeof(float);

    gpuErrchk(cudaMalloc(&d_v1x, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v1y, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v1z, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v2x, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v2y, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v2z, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v3x, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v3y, size_vert_d));
    gpuErrchk(cudaMalloc(&d_v3z, size_vert_d));
    gpuErrchk(cudaMalloc(&d_Ox, size_ray));
    gpuErrchk(cudaMalloc(&d_Oy, size_ray));
    gpuErrchk(cudaMalloc(&d_Oz, size_ray));
    gpuErrchk(cudaMalloc(&d_Dx, size_ray));
    gpuErrchk(cudaMalloc(&d_Dy, size_ray));
    gpuErrchk(cudaMalloc(&d_Dz, size_ray));
    gpuErrchk(cudaMalloc(&d_Wx, size_W));
    gpuErrchk(cudaMalloc(&d_Wy, size_W));
    gpuErrchk(cudaMalloc(&d_Wz, size_Wz));
    gpuErrchk(cudaMalloc(&d_hit_cnt, size_cnt));
    gpuErrchk(cudaMalloc(&d_iFBS, size_cnt));
    gpuErrchk(cudaMalloc(&d_Wout, size_Wout));

    // Transfer Mesh to device
    gpuErrchk(cudaMemcpy(d_v1x, &mesh.elements[0], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v1y, &mesh.elements[no_mesh], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v1z, &mesh.elements[2 * no_mesh], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v2x, &mesh.elements[3 * no_mesh], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v2y, &mesh.elements[4 * no_mesh], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v2z, &mesh.elements[5 * no_mesh], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v3x, &mesh.elements[6 * no_mesh], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v3y, &mesh.elements[7 * no_mesh], size_vert_h, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_v3z, &mesh.elements[8 * no_mesh], size_vert_h, cudaMemcpyHostToDevice));

    // Calculate the edges V1 to V2 and V1 to V3
    DestMinusOrig<<<no_mesh_d / 32, 32>>>(d_v1x, d_v1y, d_v1z, d_v2x, d_v2y, d_v2z);
    gpuErrchk(cudaPeekAtLastError());
    DestMinusOrig<<<no_mesh_d / 32, 32>>>(d_v1x, d_v1y, d_v1z, d_v3x, d_v3y, d_v3z);
    gpuErrchk(cudaPeekAtLastError());

    // Process data chunk-wise
    size_t size_current_chunk;
    for (unsigned int chnkIdx = 0; chnkIdx < no_chnk; chnkIdx++)
    {
        // The last chunk is smaller
        size_current_chunk = (chnkIdx == no_chnk - 1) ? (no_ray - chnkIdx * no_ray_chnk) * sizeof(float) : size_ray;

        if (verbose == 2)
            cout << "Processing:     Chunk " << chnkIdx << " with " << size_current_chunk / 4 << " rays." << endl;

        // Transfer ray origins and destinations to device
        gpuErrchk(cudaMemcpy(d_Ox, &orig.elements[chnkIdx * no_ray_chnk], size_current_chunk, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Oy, &orig.elements[chnkIdx * no_ray_chnk + no_ray], size_current_chunk, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Oz, &orig.elements[chnkIdx * no_ray_chnk + 2 * no_ray], size_current_chunk, cudaMemcpyHostToDevice));

        gpuErrchk(cudaMemcpy(d_Dx, &dest.elements[chnkIdx * no_ray_chnk], size_current_chunk, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Dy, &dest.elements[chnkIdx * no_ray_chnk + no_ray], size_current_chunk, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Dz, &dest.elements[chnkIdx * no_ray_chnk + 2 * no_ray], size_current_chunk, cudaMemcpyHostToDevice));

        // Calculate dest-orig on device
        DestMinusOrig<<<no_ray_chnk / 32, 32>>>(d_Ox, d_Oy, d_Oz, d_Dx, d_Dy, d_Dz);
        gpuErrchk(cudaPeekAtLastError());

        // Reset hit-counter on device
        gpuErrchk(cudaMemset(d_hit_cnt, 0, size_cnt));
        gpuErrchk(cudaMemset(d_Wout, 0, size_Wout));

        // Determine Ray-Mesh intersections
        dim3 dimGrid(no_ray_chnk, no_mesh_blk);
        RayMeshIntersect<<<dimGrid, BLOCK_SIZE_RayMesh>>>(d_v1x, d_v1y, d_v1z, d_v2x, d_v2y, d_v2z, d_v3x, d_v3y, d_v3z,
                                                          d_Ox, d_Oy, d_Oz, d_Dx, d_Dy, d_Dz, d_Wx, d_Wy, d_Wz, d_hit_cnt, d_Wout, no_hit_W);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Calculate first and last intersect points
        CalcIntersectPoints<<<no_ray_chnk / 32, 32>>>(no_mesh_blk, d_Ox, d_Oy, d_Oz, d_Dx, d_Dy, d_Dz, d_Wx, d_Wy, d_Wz, d_iFBS);
        gpuErrchk(cudaPeekAtLastError());

        // Read FBS and LBS from device
        gpuErrchk(cudaMemcpy(&fbs.elements[chnkIdx * no_ray_chnk], d_Ox, size_current_chunk, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&fbs.elements[chnkIdx * no_ray_chnk + no_ray], d_Oy, size_current_chunk, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&fbs.elements[chnkIdx * no_ray_chnk + 2 * no_ray], d_Oz, size_current_chunk, cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy(&lbs.elements[chnkIdx * no_ray_chnk], d_Dx, size_current_chunk, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&lbs.elements[chnkIdx * no_ray_chnk + no_ray], d_Dy, size_current_chunk, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&lbs.elements[chnkIdx * no_ray_chnk + 2 * no_ray], d_Dz, size_current_chunk, cudaMemcpyDeviceToHost));

        // Read hit counter and iFBS from device
        size_current_chunk = (chnkIdx == no_chnk - 1) ? (no_ray - chnkIdx * no_ray_chnk) * sizeof(unsigned int) : size_cnt;
        gpuErrchk(cudaMemcpy(&hit[chnkIdx * no_ray_chnk], d_hit_cnt, size_current_chunk, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(&iFBS[chnkIdx * no_ray_chnk], d_iFBS, size_current_chunk, cudaMemcpyDeviceToHost));

        // Read normalized intersect distances from device
        size_current_chunk = (chnkIdx == no_chnk - 1) ? no_hit_W * (no_ray - chnkIdx * no_ray_chnk) * sizeof(float) : size_Wout;
        gpuErrchk(cudaMemcpy(&Wout.elements[no_hit_W * chnkIdx * no_ray_chnk], d_Wout, size_current_chunk, cudaMemcpyDeviceToHost));

        // Update progress bar
        if (verbose == 1)
        {
            float m1 = ceil(vb_dots * ((float)chnkIdx + 1) / ((float)no_chnk));
            if (m1 > m0)
            {
                for (float m2 = 0; m2 < m1 - m0; m2++)
                    cout << "o" << flush;
                m0 = m1;
            }
        }
    }

    // Free device memory
    gpuErrchk(cudaFree(d_v1x));
    gpuErrchk(cudaFree(d_v1y));
    gpuErrchk(cudaFree(d_v1z));
    gpuErrchk(cudaFree(d_v2x));
    gpuErrchk(cudaFree(d_v2y));
    gpuErrchk(cudaFree(d_v2z));
    gpuErrchk(cudaFree(d_v3x));
    gpuErrchk(cudaFree(d_v3y));
    gpuErrchk(cudaFree(d_v3z));
    gpuErrchk(cudaFree(d_Ox));
    gpuErrchk(cudaFree(d_Oy));
    gpuErrchk(cudaFree(d_Oz));
    gpuErrchk(cudaFree(d_Dx));
    gpuErrchk(cudaFree(d_Dy));
    gpuErrchk(cudaFree(d_Dz));
    gpuErrchk(cudaFree(d_Wx));
    gpuErrchk(cudaFree(d_Wy));
    gpuErrchk(cudaFree(d_Wz));
    gpuErrchk(cudaFree(d_hit_cnt));
    gpuErrchk(cudaFree(d_iFBS));
    gpuErrchk(cudaFree(d_Wout));

    gpuErrchk(cudaDeviceReset());

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    if (verbose == 1)
        cout << "] " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds" << endl;
    else if (verbose == 2)
        cout << "Elapsed time:   " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds" << endl;

    return;
}
