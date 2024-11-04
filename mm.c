#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Resources Used: https://www.javatpoint.com/how-to-add-matrix-in-c

void MatrixMulOnHost(float* A, float* B, float* C, int Width) {
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

int main() {
	int size = 1000;
	
	float* x = malloc(sizeof(float) * size * size);
	float* y = malloc(sizeof(float) * size * size);
	float* z = malloc(sizeof(float) * size * size);

  	for (int i = 0; i < size; i++) {
    	for (int j = 0; j < size; j++) {
	      	x[i * size + j] = 1; // x[i][j]
	      	y[i * size + j] = 1;
    	}
  	}

  	 MatrixMulOnHost(x, y, z, size);
  	// printf("%d", MatrixMulOnHost(x, y, z, size));

  for (int i = 0; i < size; i++) {
  	for (int j = 0; j < size; j++) {
  		printf("%f ", z[i * size + j]);
  		if (z[i * size + j] != size) {
  			printf("Error at z[%d][%d]: %f\n", i, j, z[i * size + j]);
  		}
    }
    printf("\n");
  }

  return 0;
}
