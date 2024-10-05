#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#define SIZE 1024

void initialize_matrices(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {		    for (int i = 0; i < SIZE; i++) {
		       
	for (int j = 0; j < SIZE; j++) {

		A[i][j] = rand() % 10;
		B[i][j] = rand() % 10;
		C[i][j] = 0;
	}
}
}

void matrix_multiply(double A[SIZE][SIZE], double B[SIZE][SIZE], double C[SIZE][SIZE]) {
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			for (int k = 0; k < SIZE; k++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

int main() {
	double A[SIZE][SIZE], B[SIZE][SIZE], C[SIZE][SIZE];


	initialize_matrices(A, B, C);


	pid_t pid = fork();

	if (pid < 0) {
		perror("Fork failed");
		return 1;
	} else if (pid == 0) {
		matrix_multiply(A, B, C);
		      
		printf("Child process completed matrix multiplication.\n");
	
	} else {
		matrix_multiply(A, B, C);
		       
		printf("Parent process completed matrix multiplication.\n");
	
		wait(NULL);
	}
	return 0;
}
