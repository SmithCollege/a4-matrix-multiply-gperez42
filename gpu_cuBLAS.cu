#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#include "cublas_v2.h" 

// CUDA runtime
#include <cuda_runtime.h>
#include <sys/time.h>

#define SIZE 50

// Resources Used: https://www.javatpoint.com/how-to-add-matrix-in-c

double get_clock() {
  struct timeval tv; 
  int ok = gettimeofday(&tv, (void *) 0);
  if (ok<0) { 
  	printf("gettimeofday error"); 
  }
  return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int main() {
	 printf("[Matrix Multiply CUBLAS] - Starting...\n");

	cublasHandle_t handle;
	cublasCreate(&handle);
	
	const float alpha = 1.0f;
	const float beta = 0.0f;
       
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

  // MatrixMulOnDevice<<<1,SIZE>>>(x,y,z,size);
  cublasSgemm(
	  handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size,
      size, &alpha, x, size, y, size, &beta, z, size);

  printf("%s\n", cudaGetErrorString(cudaGetLastError()));

  // synchronize 
  cudaDeviceSynchronize();

  for (int i = 0; i < size; i++) {
  	for (int j = 0; j < size; j++) {
  		printf("%f ", z[i * size + j]);
  		if (z[i * size + j] != size) {
  			printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
  		}
    }
    printf("\n");
  }

  double t1 = get_clock();
  printf("time per call: %f ns\n", t1-t0);


  	// Freeing memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);
  

  return 0;
}
