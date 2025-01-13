// LAB 1
#include <wb.h>

__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len){
    out[i] = in1[i] + in2[i];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;

  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  // get all the data from the data folder and have pointers to them
  hostInput1 = // A CPU data
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 = // B
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  // the data we return
  hostOutput = (float *)malloc(inputLength * sizeof(float)); // C

  wbLog(TRACE, "The input length is ", inputLength); 
  int size = inputLength * sizeof(float);
  float *A_d, *B_d, *C_d;


  //@@ Allocate GPU memory here, then copy data from host to device
  cudaMalloc((void**) &A_d, size);
  cudaMalloc((void**) &B_d, size);
  cudaMalloc((void**) &C_d, size);

  //@@ Copy memory to the GPU here
  cudaMemcpy(A_d, hostInput1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, hostInput2, size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(inputLength/256.0), 1, 1);
  dim3 DimBlock(256, 1, 1);

  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, inputLength);

  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  // copy data into the hostOutput
  cudaMemcpy(hostOutput, C_d, size, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  wbSolution(args, hostOutput, inputLength);

  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
