// Histogram Equalization

#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE  512

//@@ insert code here
__global__ void float_to_unsigned(float *input, unsigned char *output, int len){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("inside of float)to_unsigned\n");
        printf("the float input is %f\n", input[index]);
    }
    if (index < len){
        output[index] = static_cast<unsigned char>(255 * input[index]);
    }
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("the unsigned char output is %u\n\n", output[index]);
    }
}

__global__ void rgb_to_grayscale(unsigned char *input, unsigned char *output, int len){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("inside of rgb to grayscale\n");
        printf("the r input is %u\n", input[3*index]);
        printf("the g input is %u\n", input[3*index+1]);
        printf("the b input is %u\n", input[3*index+2]);
    }
    if(index < len){
        unsigned char r = input[3*index];
        unsigned char g = input[3*index + 1];
        unsigned char b = input[3*index + 1];
        output[index] = static_cast<unsigned char> (0.21*r + 0.71*g + 0.07*b);
        if(threadIdx.x == 0 && blockIdx.x == 0){
            printf("the output grayscale value is %u\n\n", output[index]);
        }
    }

}

__global__ void compute_grayscale_histogram(unsigned char *grayimage, unsigned int *histogram, int len){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

    if (threadIdx.x < HISTOGRAM_LENGTH){
        histo_private[threadIdx.x] = 0;
    }
    __syncthreads();

    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("the grayscale value we are adding to the histogram is %u\n", grayimage[index]);
        printf("the histogram value at grayImage before is  %u\n", histo_private[grayimage[index]]);
    }

    if (index < len){ // maybe change this into that while loop from notes??? ---------------------------
        atomicAdd(&(histo_private[grayimage[index]]), 1);
    }
    if(threadIdx.x == 0 && blockIdx.x == 0){
        printf("the histogram value at grayImage after is  %u\n", histo_private[grayimage[index]]);
    }
    __syncthreads();

    if (threadIdx.x < HISTOGRAM_LENGTH){
        atomicAdd(&(histogram[threadIdx.x]), histo_private[threadIdx.x]);
    }

}


__global__ void cdf_of_histogram(unsigned int *histogram, float *output, int len){
    int index = threadIdx.x; // + blockIdx.x * blockDim.x;
    __shared__ float cdf[HISTOGRAM_LENGTH];
    if(index < HISTOGRAM_LENGTH){
        cdf[index] = static_cast<float>(histogram[index]); // this could possibly be wrong way to load it
    }
    __syncthreads();

    int stride = 1;
    while (stride < HISTOGRAM_LENGTH){
        __syncthreads();
        //int idx = (threadIdx.x + 1) * stride * 2 - 1;
        int idx = index;
        float tmp = 0;
        if(idx < HISTOGRAM_LENGTH && idx-stride >= 0){
            tmp = cdf[idx-stride];
        }
        __syncthreads();
        cdf[idx] += tmp;

        stride = stride * 2;
    }

    // stride = HISTOGRAM_LENGTH / 4;
    // while(stride > 0){
    //     __syncthreads();
    //     int idx = (threadIdx.x + 1) * stride * 2 - 1;
    //     if((idx + stride < HISTOGRAM_LENGTH)){
    //         cdf[idx + stride] += cdf[idx];
    //     }
    //     stride = stride / 2;
    // }
    __syncthreads();

    if (index < HISTOGRAM_LENGTH){
        output[index] = (cdf[index]) / static_cast<float>(len);
    }
}

__global__ void equalize_histogram(unsigned char *devDataCharacter, float *cdf, int len){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < len){
        float correct_color = (float)(255.0f*(cdf[devDataCharacter[index]] - cdf[0])/(1.0f - cdf[0]));
        correct_color = min(max(correct_color, 0.0f), 255.0f);
        devDataCharacter[index] = (unsigned char)correct_color;
    }
}

__global__ void unsigned_to_float(unsigned char *input, float *output, int len){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < len){
        output[index] = (float)input[index] / 255.0f;
    }
}

void printFloatArray(float* arr, int len) {
    printf("inside of the function\n");
    for (int i = 0; i < len; ++i) {
        if (i >= 0 && i < 10){
            printf("inside of the for loop\n");
            printf("%f ", arr[i]);
        }
    }
    printf("made it outside the for loop\n");
    printf("\n");
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  unsigned char * grayscale;
  unsigned int * histogram;
  float * cdf;
  float* devImageData;
  unsigned char * devDataCharacter;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  //@@ insert code here
  int image_area = imageHeight*imageWidth;
  int total_images_area = image_area * imageChannels;

  // get data for the host
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbCheck(cudaMalloc((void**) &grayscale, image_area*sizeof(unsigned char)));
  wbCheck(cudaMalloc((void**) &histogram, HISTOGRAM_LENGTH*sizeof(unsigned int)));
  wbCheck(cudaMalloc((void**) &cdf, HISTOGRAM_LENGTH*sizeof(float)));
  wbCheck(cudaMalloc((void**) &devImageData, total_images_area * sizeof(float)));
  wbCheck(cudaMalloc((void**) &devDataCharacter, total_images_area * sizeof(unsigned char)));

  wbCheck(cudaMemcpy(devImageData, hostInputImageData, total_images_area * sizeof(float), cudaMemcpyHostToDevice));
  cudaMemset(histogram, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));

//   printf("test print\n");
//   wbCheck(cudaMemcpy(hostInputImageData, devImageData, total_images_area * sizeof(float), cudaMemcpyDeviceToHost));
//   printFloatArray(hostInputImageData, total_images_area);
  dim3 DimGrid1((total_images_area + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1); 
  dim3 DimBlock1(BLOCK_SIZE, 1, 1);
  float_to_unsigned<<<DimGrid1, DimBlock1>>> (devImageData, devDataCharacter, total_images_area);
  cudaDeviceSynchronize();

  dim3 DimGrid2((image_area + (BLOCK_SIZE) - 1) / (BLOCK_SIZE), 1, 1);
  dim3 DimBlock2(BLOCK_SIZE, 1, 1);
  rgb_to_grayscale<<<DimGrid2, DimBlock2>>> (devDataCharacter, grayscale, image_area);
  cudaDeviceSynchronize();

  compute_grayscale_histogram<<<DimGrid2, DimBlock2>>> (grayscale, histogram, image_area);
  cudaDeviceSynchronize();

  dim3 DimGrid3(1, 1, 1);
  dim3 DimBlock3(HISTOGRAM_LENGTH, 1, 1);
  cdf_of_histogram<<<DimGrid3, DimBlock3>>> (histogram, cdf, image_area);
  cudaDeviceSynchronize();

  equalize_histogram<<<DimGrid1, DimBlock1>>>(devDataCharacter, cdf, total_images_area);
  cudaDeviceSynchronize();

  unsigned_to_float<<<DimGrid1, DimBlock1>>>(devDataCharacter, devImageData, total_images_area);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, devImageData, total_images_area * sizeof(float), cudaMemcpyDeviceToHost); 

  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(devDataCharacter);
  cudaFree(devImageData);
  cudaFree(grayscale);
  cudaFree(histogram);
  cudaFree(cdf);

  return 0;
}

