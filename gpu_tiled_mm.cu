#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

#define SIZE 128
#define TILE_WIDTH 2

// Resources Used: https://www.javatpoint.com/how-to-add-matrix-in-c

/*
__global__ void MatrixMulOnDevice(float* A, float* B, float* C, int Width) {
	 for (int i = 0; i < Width; ++i){
		 for (int j = 0; j < Width; ++j) {
			 float sum = 0;
			 for (int k = 0; k < Width; ++k) {
				 float a = A[i * Width + k];
				 float b = B[k * Width + j];
				 sum += a * b;
			 }
		 C[i * Width + j] = sum;

		 }
	 }
}
*/
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {

	__shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	// Loop over the M and N tiles required to compute the P element
	// The code assumes that the Width is a multiple of TILE_WIDTH!
	
	for (int m = 0; m < Width/TILE_WIDTH; ++m) {
		// Collaborative loading of M and N tiles into shared memory
		subTileM[ty][tx] = M[Row*Width + m*TILE_WIDTH+tx];
		subTileN[ty][tx] = N[(m*TILE_WIDTH+ty)*Width+Col];
		 __syncthreads();
		 
		for (int k = 0; k < TILE_WIDTH; ++k)
		 Pvalue += subTileM[ty][k] * subTileN[k][tx];
		 __syncthreads();

	 P[Row*Width+Col] = Pvalue;
	}
}

double get_clock() {
  struct timeval tv; 
  int ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { 
  	printf("gettimeofday error"); 
  }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
	int size = 100;
	float *x, *y, *z;

	double t0 = get_clock();

	cudaMallocManaged(&x, SIZE*sizeof(float) * size * size);
	cudaMallocManaged(&y, SIZE*sizeof(float) * size * size);
	cudaMallocManaged(&z, SIZE*sizeof(float) * size * size);

  	for (int i = 0; i < size; i++) {
    	for (int j = 0; j < size; j++) {
	      	x[i * size + j] = 1; // x[i][j]
	      	y[i * size + j] = 1;
    	}
  	}

  dim3 dimGrid(ceil((1.0*size)/TILE_WIDTH),
  ceil((1.0*size)/TILE_WIDTH), 1);
   dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

   MatrixMulKernel<<<dimGrid, dimBlock>>>(x, y, z, size);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  // synchronize 
  cudaDeviceSynchronize();

  for (int i = 0; i < size; i++) {
  	for (int j = 0; j < size; j++) {
  		printf("%f ", z[i * size + j]);
  		if (z[i * size + j] != size) {
  			// printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
  		}
    }
    printf("\n");
  }

   double t1 = get_clock();
    printf("time per call: %f s\n", t1-t0);

  	// Freeing memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
  

  return 0;
}
