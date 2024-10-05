#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
    int m, n, k;
    double *a;
    int lda;
    double *b;
    int ldb;
    double *c;
    int ldc;
    int block_size;
    int start_row;
    int end_row;
} thread_args;

void *multiply_block(void *arg) {
    thread_args *args = (thread_args *)arg;
    int m = args->m, n = args->n, k = args->k;
    int block_size = args->block_size;
    double *a = args->a, *b = args->b, *c = args->c;
    int lda = args->lda, ldb = args->ldb, ldc = args->ldc;
    int start_row = args->start_row, end_row = args->end_row;

    for (int i = start_row; i < end_row; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int ii = i; ii < i + block_size && ii < m; ++ii) {
                for (int jj = j; jj < j + block_size && jj < n; ++jj) {
                    double sum = 0;
                    for (int kk = 0; kk < k; ++kk) {
                        sum += a[ii * lda + kk] * b[kk * ldb + jj];
                    }
                    c[ii * ldc + jj] += sum;
                }
            }
        }
    }
    pthread_exit(NULL);
}

void MY_MMult(int m, int n, int k, double *a, int lda,
              double *b, int ldb, double *c, int ldc) {
    int num_threads = 4;
    int block_size = 2; // Example block size
    pthread_t threads[num_threads];
    thread_args args[num_threads];

    int rows_per_thread = m / num_threads;
    for (int i = 0; i < num_threads; i++) {
        args[i].m = m;
        args[i].n = n;
        args[i].k = k;
        args[i].a = a;
        args[i].lda = lda;
        args[i].b = b;
        args[i].ldb = ldb;
        args[i].c = c;
        args[i].ldc = ldc;
        args[i].block_size = block_size;
        args[i].start_row = i * rows_per_thread;
        args[i].end_row = (i == num_threads - 1) ? m : (i + 1) * rows_per_thread;

        pthread_create(&threads[i], NULL, multiply_block, (void *)&args[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
}
