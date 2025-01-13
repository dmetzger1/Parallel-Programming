#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K){ //, const int offset) {
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
    size_t t = blockIdx.x * BLOCK_SIZE + threadIdx.x; //+ offset; // # add in another test case to make sure we're not out of the bounds for the stream???
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
        for(int p = 0; p < K; p++){     // for each element of the KxK filter (two loops)
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
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, 
                                                    const float *host_mask, float **device_output_ptr, 
                                                    float **device_input_ptr, float **device_mask_ptr, 
                                                    const int Batch, const int Map_out, 
                                                    const int Channel, const int Height, 
                                                    const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    int num_streams = 10;
    // cudaStream_t stream_array[num_streams];
    // for (int i = 0; i < num_streams; i++){
    //     cudaStreamCreate(&stream_array[i]);
    // }
    cudaStream_t stream_1;
    cudaStreamCreate(&stream_1);
    cudaStream_t stream_2;
    cudaStreamCreate(&stream_2);
    cudaStream_t stream_3;
    cudaStreamCreate(&stream_3);


    size_t Height_out = Height - K + 1;
    size_t Width_out = Width - K + 1;
    size_t Height_unrolled = Channel * K * K;
    size_t Width_unrolled = Batch * Height_out * Width_out;
    size_t batches_per_stream = ceil((Batch *1.0)/num_streams);

    size_t input_length = Batch * Channel * Width * Height;
    size_t unrolled_matrix_length = Batch * Channel * K * K * Height_out * Width_out;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);

    // int seg_number = 10;
    size_t unrolled_matrix_stream_length = ceil((unrolled_matrix_length *1.0) / num_streams);
    // int output_seg_size = ceil((output_size*1.0) / seg_number);

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    float *host_output_pin;
    float *host_input_pin;

    cudaMallocHost((void**) &host_input_pin, input_length * sizeof(float));

    cudaMallocHost((void**) &host_output_pin, output_size);

    memcpy(host_input_pin, host_input, input_length * sizeof(float));

    cudaMalloc((void**) device_input_ptr, input_length * sizeof(float));

    cudaMalloc((void**) device_output_ptr, output_size);

    cudaMalloc((void**) device_mask_ptr, mask_size);

    cudaMalloc((void**)&unrolled_matrix, unrolled_matrix_length * sizeof(float));

    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // cudaMemcpy(*device_input_ptr, host_input_pin, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    float *device_output = *device_output_ptr;
    float *device_input = *device_input_ptr;
    float *device_mask = *device_mask_ptr;

    size_t num_threads = Channel * Height_out * Width_out;
    size_t numBlocks = ((num_threads + BLOCK_SIZE - 1)/BLOCK_SIZE);

    size_t width_unrolled_per_stream = Height_out * Width_out * batches_per_stream;

 ////////////////////////////////////////////
    for (int i = 0; i < 4; i++){
        size_t input_offset_1 = 3*i * batches_per_stream * Channel * Width * Height;
        size_t input_offset_2 = (3*i + 1) * batches_per_stream * Channel * Width * Height;
        size_t input_offset_3 = (3*i + 2) * batches_per_stream * Channel * Width * Height;

        size_t input_stream_length = batches_per_stream * Channel * Width * Height;

        size_t unrolled_offset_1 = 3*i * batches_per_stream * Channel * K * K * Height_out * Width_out;
        size_t unrolled_offset_2 = (3*i + 1) * batches_per_stream * Channel * K * K * Height_out * Width_out;
        size_t unrolled_offset_3 = (3*i + 2) * batches_per_stream * Channel * K * K * Height_out * Width_out;

        size_t matmul_offset_1 = 3*i * batches_per_stream * Map_out * Height_out * Width_out; 
        size_t matmul_offset_2 = (3*i + 1) * batches_per_stream * Map_out * Height_out * Width_out; 
        size_t matmul_offset_3 = (3*i + 2) * batches_per_stream * Map_out * Height_out * Width_out;

        size_t output_offset_1 = 3*i * batches_per_stream * Map_out * Height_out * Width_out;
        size_t output_offset_2 = (3*i + 1) * batches_per_stream * Map_out * Height_out * Width_out;
        size_t output_offset_3 = (3*i + 2) * batches_per_stream * Map_out * Height_out * Width_out;
        
        // copy from host top device for both streams
        cudaMemcpyAsync(device_input + input_offset_1, host_input_pin + input_offset_1, input_stream_length * sizeof(float), cudaMemcpyHostToDevice, stream_1);
        if(i < 3){
            cudaMemcpyAsync(device_input + input_offset_2, host_input_pin + input_offset_2, input_stream_length * sizeof(float), cudaMemcpyHostToDevice, stream_2);
            cudaMemcpyAsync(device_input + input_offset_3, host_input_pin + input_offset_3, input_stream_length * sizeof(float), cudaMemcpyHostToDevice, stream_3);
        }

        dim3 blockDim1(BLOCK_SIZE, 1, 1);
        dim3 gridDim1(numBlocks, batches_per_stream, 1);

        dim3 gridDim2(((width_unrolled_per_stream + TILE_WIDTH -1) / TILE_WIDTH), ((Height_unrolled + TILE_WIDTH - 1)/TILE_WIDTH), 1);
        dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

        size_t out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, batches_per_stream, 1);

        // do kernels for stream 1
        matrix_unrolling_kernel<<<gridDim1, blockDim1, 0, stream_1>>>(&device_input[input_offset_1], &unrolled_matrix[unrolled_offset_1], batches_per_stream, Channel, Height, Width, K);
        matrixMultiplyShared<<<gridDim2, blockDim2, 0, stream_1>>>(device_mask, &unrolled_matrix[unrolled_offset_1], &matmul_output[matmul_offset_1],
                                     Map_out, Height_unrolled,
                                     Height_unrolled, width_unrolled_per_stream,
                                     Map_out, width_unrolled_per_stream);
        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream_1>>>(&matmul_output[matmul_offset_1], &device_output[output_offset_1], Map_out, batches_per_stream, out_image_size);

       if (i < 3){
            // do kernels for stream 2
            matrix_unrolling_kernel<<<gridDim1, blockDim1, 0, stream_2>>>(&device_input[input_offset_2], &unrolled_matrix[unrolled_offset_2], batches_per_stream, Channel, Height, Width, K);
            matrixMultiplyShared<<<gridDim2, blockDim2, 0, stream_2>>>(device_mask, &unrolled_matrix[unrolled_offset_2], &matmul_output[matmul_offset_2],
                                        Map_out, Height_unrolled,
                                        Height_unrolled, width_unrolled_per_stream,
                                        Map_out, width_unrolled_per_stream);
            matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream_2>>>(&matmul_output[matmul_offset_2], &device_output[output_offset_2], Map_out, batches_per_stream, out_image_size);

            // do kernels for stream 3
            matrix_unrolling_kernel<<<gridDim1, blockDim1, 0, stream_3>>>(&device_input[input_offset_3], &unrolled_matrix[unrolled_offset_3], batches_per_stream, Channel, Height, Width, K);
            matrixMultiplyShared<<<gridDim2, blockDim2, 0, stream_3>>>(device_mask, &unrolled_matrix[unrolled_offset_3], &matmul_output[matmul_offset_3],
                                        Map_out, Height_unrolled,
                                        Height_unrolled, width_unrolled_per_stream,
                                        Map_out, width_unrolled_per_stream);
            matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream_3>>>(&matmul_output[matmul_offset_3], &device_output[output_offset_3], Map_out, batches_per_stream, out_image_size);
       }

        // copy back from device to host on both streams
        size_t output_stream_length = batches_per_stream * Map_out * Height_out * Width_out;
        cudaMemcpyAsync(&host_output_pin[output_offset_1], &device_output[output_offset_1], output_stream_length * sizeof(float), cudaMemcpyDeviceToHost, stream_1);
        if (i < 3){
            cudaMemcpyAsync(&host_output_pin[output_offset_2], &device_output[output_offset_2], output_stream_length * sizeof(float), cudaMemcpyDeviceToHost, stream_2);
            cudaMemcpyAsync(&host_output_pin[output_offset_3], &device_output[output_offset_3], output_stream_length * sizeof(float), cudaMemcpyDeviceToHost, stream_3);
        }
    }


////////////////////////////////////////

    // for (int i = 0; i < num_streams; i++){
    //     size_t input_offset = i * batches_per_stream * Channel * Width * Height;
    //     size_t input_stream_length = batches_per_stream * Channel * Width * Height;
    //     cudaMemcpyAsync(device_input + input_offset, host_input_pin + input_offset, input_stream_length * sizeof(float), cudaMemcpyHostToDevice, stream_array[i]);
    // }

    // // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    // for (int i = 0; i < num_streams; i++){
    //     size_t input_offset = i * batches_per_stream * Channel * Height * Width;
    //     size_t unrolled_offset = i * batches_per_stream * Channel * K * K * Height_out * Width_out;
    //     size_t matmul_offset = i * batches_per_stream * Map_out * Height_out * Width_out; 
    //     size_t output_offset = i * batches_per_stream * Map_out * Height_out * Width_out;

    //     dim3 blockDim1(BLOCK_SIZE, 1, 1);
    //     dim3 gridDim1(numBlocks, batches_per_stream, 1);

    //     dim3 gridDim2(((width_unrolled_per_stream + TILE_WIDTH -1) / TILE_WIDTH), ((Height_unrolled + TILE_WIDTH - 1)/TILE_WIDTH), 1);
    //     dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

    //     matrix_unrolling_kernel<<<gridDim1, blockDim1, 0, stream_array[i]>>>(&device_input[input_offset], &unrolled_matrix[unrolled_offset], batches_per_stream, Channel, Height, Width, K);// , input_offset);
    //     //cudaDeviceSynchronize();

    //     matrixMultiplyShared<<<gridDim2, blockDim2, 0, stream_array[i]>>>(device_mask, &unrolled_matrix[unrolled_offset], &matmul_output[matmul_offset],
    //                                  Map_out, Height_unrolled,
    //                                  Height_unrolled, width_unrolled_per_stream,
    //                                  Map_out, width_unrolled_per_stream);
    //     //cudaDeviceSynchronize();


    //     size_t out_image_size = Height_out * Width_out;
    //     dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, batches_per_stream, 1);
    //     matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(&matmul_output[matmul_offset], &device_output[output_offset], Map_out, batches_per_stream, out_image_size);
    //     //cudaDeviceSynchronize();

    // }
    // // matrix_unrolling_kernel<<<gridDim1, blockDim1>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);



    // for (int i = 0; i < num_streams; i++){
    //     size_t output_offset = i * batches_per_stream * Map_out * Height_out * Width_out;
    //     size_t output_stream_length = batches_per_stream * Map_out * Height_out * Width_out;
    //     cudaMemcpyAsync(&host_output_pin[output_offset], &device_output[output_offset], output_stream_length * sizeof(float), cudaMemcpyDeviceToHost, stream_array[i]);
    // }


    // TODO: Set the kernel dimensions and call the matmul kernel
    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    cudaStreamDestroy(stream_3);

    // cudaMemcpy(host_output_pin, device_output, output_size, cudaMemcpyDeviceToHost);
    memcpy((void*)host_output, host_output_pin, output_size);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
    cudaFreeHost(host_input_pin);
    cudaFreeHost(host_output_pin);


    //Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error 5: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // const int Height_unrolled = Channel * K * K;
    // const int Width_unrolled = Batch * Height_out * Width_out;

    // int num_threads = Channel * Height_out * Width_out;
    // int numBlocks = ((num_threads + BLOCK_SIZE - 1)/BLOCK_SIZE);

    // float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    // float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    // cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    // cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    // dim3 blockDim1(BLOCK_SIZE, 1, 1);
    // dim3 gridDim1(numBlocks, Batch, 1);

    // matrix_unrolling_kernel<<<gridDim1, blockDim1>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    // cudaDeviceSynchronize();

    // // TODO: Set the kernel dimensions and call the matmul kernel
    // dim3 gridDim2(((Width_unrolled + TILE_WIDTH -1) / TILE_WIDTH), ((Height_unrolled + TILE_WIDTH - 1)/TILE_WIDTH), 1);
    // dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);

    // matrixMultiplyShared<<<gridDim2, blockDim2>>>(device_mask, unrolled_matrix, matmul_output,
    //                                  Map_out, Height_unrolled,
    //                                  Height_unrolled, Width_unrolled,
    //                                  Map_out, Width_unrolled);
    // cudaDeviceSynchronize();

    // // Permute the result of matrix multiplication
    // const int out_image_size = Height_out * Width_out;
    // dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
    //     matmul_output, device_output, Map_out, Batch, out_image_size
    // );

    // cudaFree(matmul_output);
    // cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // // Copy the output back to host
    // int Height_out = Height - K + 1;
    // int Width_out = Width - K + 1;
    // int output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

    // cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // // Free device memory
    // cudaFree(device_input);
    // cudaFree(device_output);
    // cudaFree(device_mask);

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