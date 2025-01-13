// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[2*BLOCK_SIZE];
  int i = blockIdx.x * blockDim.x *2;
  if (threadIdx.x + i < len){
    XY[threadIdx.x] = input[threadIdx.x + i];
  }
  if (threadIdx.x + i + blockDim.x < len){
    XY[threadIdx.x + blockDim.x] = input[threadIdx.x + i + blockDim.x];
  }

  int stride = 1;
  while(stride < 2*BLOCK_SIZE) {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0){
      XY[index] += XY[index-stride];
    }
    stride = stride*2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride > 0) {
    __syncthreads();
    int idx = (threadIdx.x+1)*stride*2 - 1;
    if ((idx + stride) < 2*BLOCK_SIZE){
      XY[idx + stride] += XY[idx];
    }
    stride = stride / 2;
  }
  if (threadIdx.x + i < len){
    output[threadIdx.x + i] = XY[threadIdx.x];
  }
  if (threadIdx.x + i + blockDim.x < len){
    output[threadIdx.x + i + blockDim.x] = XY[threadIdx.x + blockDim.x];
  }
}

__global__ void add_sums(float * sums, float * input, int len){
  int si = 2* (blockIdx.x + 1) * blockDim.x;
  int i = si + threadIdx.x;
  if( i < len){
    input[i] += sums[blockIdx.x];
  }
  if (i + blockDim.x < len){
    input[i + blockDim.x] += sums[blockIdx.x];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(((numElements - 1)/(BLOCK_SIZE*2) + 1), 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float), cudaMemcpyDeviceToHost);

  float * dev_sum_input;
  float * dev_sum_output;
  float * host_sum_input;

  cudaMalloc((void **)&dev_sum_input, ((numElements - 1)/(BLOCK_SIZE*2) + 1) * sizeof(float));
  cudaMalloc((void **)&dev_sum_output, ((numElements - 1)/(BLOCK_SIZE*2) + 1) * sizeof(float));
  host_sum_input = (float *)malloc(((numElements - 1)/(BLOCK_SIZE*2) + 1) * sizeof(float));
  // hard copy
  for (int j = 0; j < ((numElements - 1)/(BLOCK_SIZE*2) + 1); j++) {
    host_sum_input[j] = hostOutput[(j+1)*2*BLOCK_SIZE-1];
  }
  cudaMemcpy(dev_sum_input, host_sum_input, ((numElements - 1)/(BLOCK_SIZE*2) + 1) * sizeof(float), cudaMemcpyHostToDevice);

  scan<<<1, (((numElements - 1)/(BLOCK_SIZE*2) + 1)-1)/2 + 1>>>(dev_sum_input, dev_sum_output, ((numElements - 1)/(BLOCK_SIZE*2) + 1));
  add_sums<<<DimGrid, DimBlock>>>(dev_sum_output, deviceOutput, numElements);
  cudaDeviceSynchronize();

  wbTime_start(Copy, "Copying output memory to the CPU");
  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");


  //@@  Free GPU Memory
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");
  // cudaFree(dev_sum_input);
  // cudaFree(dev_sum_output);

  wbSolution(args, hostOutput, numElements);

  
  free(hostInput);
  free(hostOutput);
  // free(host_sum_input);
  return 0;
}

