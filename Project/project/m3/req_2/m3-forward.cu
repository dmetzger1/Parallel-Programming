#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256


__global__ void fusion_kernel(const float * device_input,
                                  const float * device_mask,
                                  float * device_output,
                                  size_t Batch, size_t Map_out, size_t Channel,
                                  size_t Height, size_t Width, size_t K)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // calculate dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int numARows = Map_out; // A = device_mask --> dimension: (Map_out) x (Channel*K*K)
    const int numAColumns = Channel * K * K;

    const int numBRows = Channel * K * K;  // B = unrolled input --> dimension: (Channel*K*K) x (Batch * Height_out * Width_out)
    const int numBColumns = Batch * Height_out * Width_out;

    const int numCRows = numARows; // C = device_output
    const int numCColumns = numBColumns;

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int C_row = by * TILE_WIDTH + ty;
    int C_col = bx * TILE_WIDTH + tx;

    float C_val = 0.0f;
    for (int tileId = 0; tileId < (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH; tileId++) {

        size_t B_row = tileId * TILE_WIDTH + ty; 
        size_t B_col = C_col;                       
        float B_val = 0.0f;

        size_t mask_channel_idx, cur_channel, mask_idx, p, q, cur_batch, pixel, h_out_idx, w_out_idx, h_in, w_in;
        size_t W_unroll_per_matrix = Height_out * Width_out;
    
        if (B_row < numBRows && B_col < numBColumns) {
            // Decompose B_row into channel and kernel offsets
            mask_channel_idx = B_row; // Combined index for Channel*K*K
            cur_channel = mask_channel_idx / (K*K);
            mask_idx = mask_channel_idx % (K*K);
            p = mask_idx / K;
            q = mask_idx % K;

            // Decompose B_col into batch, h_out, w_out
            cur_batch = B_col / W_unroll_per_matrix;
            pixel = B_col % W_unroll_per_matrix;
            h_out_idx = pixel / Width_out;
            w_out_idx = pixel % Width_out;

            // Ensure indices are within input boundaries
            h_in = h_out_idx + p;
            w_in = w_out_idx + q;
            if (cur_batch < Batch && cur_channel < Channel && h_in < Height && w_in < Width) {
                B_val = device_input[cur_batch * (Channel * Height * Width) +
                                     cur_channel * (Height * Width) +
                                     h_in * Width +
                                     w_in];
            }
        }
        tileB[ty][tx] = B_val;

        // Load tile A from mask as you would normally
        if (C_row < numARows && (tileId * TILE_WIDTH + tx) < numAColumns) {
            tileA[ty][tx] = device_mask[C_row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial result
        if (C_row < numCRows && C_col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                C_val += tileA[ty][i] * tileB[i][tx];
            }
        }

        __syncthreads();
    }

    size_t W_unroll_per_matrix = Height_out * Width_out;
    size_t cur_batch, pixel, h_out, w_out, cur_row;
    // Write final result in the permuted shape
    if (C_row < numCRows && C_col < numCColumns) {
        cur_batch = C_col / (Height_out * Width_out);
        pixel = C_col % (Height_out * Width_out);
        h_out = pixel / Width_out;
        w_out = pixel % Width_out;
        cur_row = C_row;

        device_output[cur_batch * (Map_out * Height_out * Width_out) +
                      cur_row * (Height_out * Width_out) +
                      h_out * Width_out +
                      w_out] = C_val;
    }
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int input_size = Batch * Channel * Width * Height * sizeof(float);
    int output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    int mask_size = Map_out * Channel * K * K * sizeof(float);

    cudaMalloc((void**) device_input_ptr, input_size);
    cudaMalloc((void**) device_output_ptr, output_size);
    cudaMalloc((void**) device_mask_ptr, mask_size);

    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, 
                                            const float *device_mask, const int Batch, 
                                            const int Map_out, const int Channel, 
                                            const int Height, const int Width, const int K)
{

    // output dimensions
    size_t Height_out = Height - K + 1;
    size_t Width_out = Width - K + 1;

    // Find dimensions for the matrix multiplication
    size_t numARows = Map_out;
    size_t numAColumns = Channel * K * K;
    size_t numBColumns = Batch * Height_out * Width_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((numBColumns + TILE_WIDTH - 1) / TILE_WIDTH, (numARows + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    fusion_kernel<<<gridDim, blockDim>>>(device_input, device_mask, device_output,
                                             Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();


/*
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    int num_threads = Channel * Height_out * Width_out;
    int numBlocks = ((num_threads + BLOCK_SIZE - 1)/BLOCK_SIZE);

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 blockDim1(BLOCK_SIZE, 1, 1);
    dim3 gridDim1(numBlocks, Batch, 1);

    matrix_unrolling_kernel<<<gridDim1, blockDim1>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    cudaDeviceSynchronize();

    // TODO: Set the kernel dimensions and call the matmul kernel
    dim3 gridDim2(((Width_unrolled + TILE_WIDTH -1) / TILE_WIDTH), ((Height_unrolled + TILE_WIDTH - 1)/TILE_WIDTH), 1);
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

    matrixMultiplyShared<<<gridDim2, blockDim2>>>(device_mask, unrolled_matrix, matmul_output,
                                     Map_out, Height_unrolled,
                                     Height_unrolled, Width_unrolled,
                                     Map_out, Width_unrolled);
    cudaDeviceSynchronize();

    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
    */
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}