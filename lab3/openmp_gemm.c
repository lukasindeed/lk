#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
void naive_gemm(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc) {

        #pragma omp parallel for
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			double sum = 0.0;
			for (int p = 0; p < k; ++p) {
				sum += a[i * lda + p] * b[p * ldb + j];
			}
			c[i * ldc + j] = sum;
		}
	}
}

