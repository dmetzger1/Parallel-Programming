// LAB 2 FA24

#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)


// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  // BlockDim.x  = the entire x size of a block
  // blockIdx.x = which x idx block it is on the grid
  // threadidx.x is the x coord of teh thread in the block
  int i = blockIdx.x * blockDim.x + threadIdx.x; // column in matrix C of the unit we are trying to calculate
  int j = blockIdx.y * blockDim.y + threadIdx.y; // row in matrix C of the unit we are trying to calculate
  if(i >= numCColumns || j >= numCRows){
    return;
  }else{
    float total = 0;
    for(int w = 0; w < numAColumns; w++){
      float a = A[j * numAColumns + w];
      float b = B[w * numBColumns + i];
      total += a*b;
    }
    C[j * numCColumns + i] = total;
    return;
  }
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  //@@ Set numCRows and numCColumns
  
  numCRows = numARows;
  numCColumns = numBColumns;

  int A_size = numAColumns*numARows * sizeof(float);
  int B_size = numBColumns*numBRows * sizeof(float);
  int C_size = numCColumns*numCRows * sizeof(float);

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(C_size); // C
  float *A_d, *B_d, *C_d;

  //@@ Allocate GPU memory here
  cudaMalloc((void**) &A_d, A_size);
  cudaMalloc((void**) &B_d, B_size);
  cudaMalloc((void**) &C_d, C_size);

  //@@ Copy memory to the GPU here
  cudaMemcpy(A_d, hostA, A_size, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, hostB, B_size, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns/16.0), ceil(numCRows/16.0), 1);  // have one block for every 16 rows, and also every 16 columns in the resulting matrix
  dim3 DimBlock(16, 16, 1); // thread block limited in size to 1024 threads, here we have 16x16x1 = 256 threads in one threadblock


  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);


  cudaDeviceSynchronize();
  
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, C_d, C_size, cudaMemcpyDeviceToHost);

  //@@ Free the GPU memory here
  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);

  return 0;
}

