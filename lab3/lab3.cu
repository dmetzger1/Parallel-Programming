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

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];

  int block_idx_x = blockIdx.x; int block_idx_y = blockIdx.y;
  int thread_idx_x = threadIdx.x; int thread_idx_y = threadIdx.y;

  // find the column and width we're working with for this thread
  int row = block_idx_y * TILE_WIDTH + thread_idx_y;
  int col = block_idx_x * TILE_WIDTH + thread_idx_x;

  float C_value = 0;

  // Loop over the M and N tiles required to compute the P element
  for (int q=0; q < (numAColumns - 1)/TILE_WIDTH + 1; q++){
    // Collaborative loading of M and N tiles into shared memory
    if(row < numARows && (q * TILE_WIDTH + thread_idx_x) < numAColumns){
      subTileA[thread_idx_y][thread_idx_x] = A[row * numAColumns + q*TILE_WIDTH+thread_idx_x];
    }else{
      subTileA[thread_idx_y][thread_idx_x] = 0;
    }
    if ((q * TILE_WIDTH + thread_idx_y) < numBRows && col < numBColumns){
      subTileB[thread_idx_y][thread_idx_x] = B[(q*TILE_WIDTH+thread_idx_y)*numBColumns + col];
    }else{
      subTileB[thread_idx_y][thread_idx_x] = 0;
    }

    __syncthreads();
    if (row < numCRows && col < numCColumns){
      for (int k=0; k < TILE_WIDTH; k++){
        C_value += subTileA[thread_idx_y][k] * subTileB[k][thread_idx_x];
      }
    }
    __syncthreads();
  }
  if (row < numCRows && col < numCColumns){
    C[row*numCColumns + col] = C_value;
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
  printf("Dimension of matrix A is %d by %d\n", numARows, numAColumns);
  printf("Dimension of matrix B is %d by %d\n", numBRows, numBColumns);
  printf("Dimension of matrix C is %d by %d\n", numCRows, numCColumns);

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil(numCColumns/(TILE_WIDTH*1.0)), ceil(numCRows/(TILE_WIDTH*1.0)), 1);  // have one block for every 16 rows, and also every 16 columns in the resulting matrix
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1); // thread block limited in size to 1024 threads, here we have 16x16x1 = 256 threads in one threadblock

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

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

  //@@ Free the hostC matrix
  free(hostC);
  return 0;
}
