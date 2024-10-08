#include <stdio.h>

#define M 3
#define N 3
#define K 2


void dgemm(int m, int n, int k, double alpha, double beta,
          double* A, double* B, double* C_naive){
    for(int i=0; i< m;i ++){    //C[i] 
        for(int j=0; j< n; j++){  //C[i][j]
            for(int p=0; p< k; p++){  
                C_naive[i*n + j] += A[i][p]*B[p][j]; 
             }
	    C_naive[i*n + j] = C_naive[i*n + j] * alpha + beta;
        }
    }
}

void printf_matrix(int row, int col, double matrix[row][col] ){
  for(int i=0; i<row; i++){
    for(int j=0; j<col;j++){
        printf("%lf ", matrix[i][j]);
    }
    printf("\n");
  }
  printf("\n\n");
}


void main()
{
    double A[M][K] = { {1.0,  -3.0},
                      {2.0,  4.0},
                      {1.0, -1.0} };         
    double B[K][N] = { {1.0,  2.0,  1.0},
                      {-3.0, 4.0, -1.0} };  
    double C[M][N] = { {.5, .5, .5}, 
                      {.5, .5, .5},
                      {.5, .5, .5} }; 

    //C=A*B + beta*C
    dgemm(M,N,K,2, A,B,C);

    printf_matrix(M,K,A);
    printf_matrix(K,N,B);
    printf_matrix(M,N,C);

}


