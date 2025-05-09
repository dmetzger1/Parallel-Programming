#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 4
#define BLOCK_WIDTH (MASK_WIDTH+TILE_WIDTH-1)

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  // int input_tile_width = TILE_WIDTH + MASK_WIDTH - 1;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  __shared__ float shared_memory[BLOCK_WIDTH][BLOCK_WIDTH][BLOCK_WIDTH];

  int row_o = blockIdx.y * TILE_WIDTH + ty;
  int col_o = blockIdx.x * TILE_WIDTH + tx;
  int dep_o = blockIdx.z * TILE_WIDTH + tz;

  int row_i = row_o - 1;
  int col_i = col_o - 1; // setting back to the correct locations of the overall mask
  int dep_i = dep_o - 1;

  float output_value = 0.0f;

  if((row_i >= 0) && (row_i < y_size) &&  // inside range of the input size
    (col_i >= 0) && (col_i < x_size) && 
      (dep_i >= 0) && (dep_i < z_size)){
        shared_memory[tz][ty][tx] = input[dep_i*y_size*x_size + row_i*x_size + col_i];
  }else{
    shared_memory[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  if(ty < TILE_WIDTH && tx < TILE_WIDTH && tz < TILE_WIDTH){
    for(int i = 0; i < MASK_WIDTH; i++){
      for(int j = 0; j < MASK_WIDTH; j++){
        for(int k = 0; k < MASK_WIDTH; k++){
          output_value += deviceKernel[i][j][k] * shared_memory[i + tz][j + ty][k + tx];
        }
      }
    }
    if(row_o < y_size && col_o < x_size && dep_o < z_size){
      output[dep_o*y_size*x_size + row_o*x_size + col_o] = output_value;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  int input_size = (inputLength-3) * sizeof(float);
  int kernel_size = kernelLength * sizeof(float);
  int output_size = input_size;

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, input_size);
  cudaMalloc((void**) &deviceOutput, output_size);

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, input_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernel_size, 0, cudaMemcpyHostToDevice);


  //@@ Initialize grid and block dimensions here
  int input_tile_width = TILE_WIDTH + MASK_WIDTH - 1;
  dim3 DimBlock(input_tile_width, input_tile_width, input_tile_width); 
  dim3 DimGrid(ceil(x_size / (TILE_WIDTH*1.0)), ceil(y_size / (TILE_WIDTH*1.0)), ceil(z_size / (TILE_WIDTH*1.0))); 

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();


  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, output_size, cudaMemcpyDeviceToHost);


  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

