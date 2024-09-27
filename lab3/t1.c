#include <stdio.h>
#include <omp.h>

#define M 1024
#define N 1024
#define K 1024

void dgemm(double A[M][K], double B[K][N], double C[M][N]) {
#pragma omp parallel for
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			double sum = 0.0;
			for (int k = 0; k < K; k++) {
				sum += A[i][k] * B[k][j];
			}
			C[i][j] = sum;
		}
	}
}

int main() {
	double A[M][K], B[K][N], C[M][N];

	for (int i = 0; i < M; i++)
		for (int k = 0; k < K; k++)
			A[i][k] = i + k;

	for (int k = 0; k < K; k++)
		for (int j = 0; j < N; j++)
			B[k][j] = k + j;

		        dgemm(A, B, C);

		
		    	printf("C[0][0] = %f\n", C[0][0]);

		
			return 0;
}

