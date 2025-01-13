#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    // each thread computes K* K number of elements
    size_t cur_channel, s, h_out_idx, w_out_idx, w_base, w_unroll_idx;
    size_t t = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int batch_num = blockIdx.y;
    size_t W_unroll_per_matrix = Height_out * Width_out; // the width of one unrolled matrix as a whole for one batch

    if(t < Channel * W_unroll_per_matrix){
        cur_channel = t / W_unroll_per_matrix;
        s = t%W_unroll_per_matrix;
        h_out_idx = s/Width_out;
        w_out_idx = s%Width_out;
        w_unroll_idx = h_out_idx * Width_out + w_out_idx;
        w_base = cur_channel * K * K;
        size_t h_unroll_idx = 0;
        #pragma unroll
        for(int p = 0; p < K; p++){     // for each element of the KxK filter (two loops)
            #pragma unroll
            for(int q = 0; q < K; q++){
                h_unroll_idx = w_base + p*K + q;
                // int input_val = in_4d(batch_num, cur_channel, h_out_idx+p, w_out_idx+q); // get input pixel value
                // int output_index = W_unroll_per_matrix * Batch * h_unroll_idx + batch_num * W_unroll_per_matrix + w_unroll_idx;
                output[(size_t)(W_unroll_per_matrix * Batch * h_unroll_idx + batch_num * W_unroll_per_matrix + w_unroll_idx)] = in_4d(batch_num, cur_channel, h_out_idx+p, w_out_idx+q);    // copy input pixels, how do i do this?
            }
        }

    }

    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            #pragma unroll
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        #pragma unroll
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
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


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
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