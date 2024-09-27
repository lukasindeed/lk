#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define M 8  // A的行数
#define K 8  // A的列数和B的行数
#define N 8  // B的列数
#define BLOCK_SIZE 2  // 分块大小

void blockMatrixMultiply(double *A, double *B, double *C, int blockSize) {
    for (int i = 0; i < blockSize; i++) {
        for (int j = 0; j < blockSize; j++) {
            double sum = 0.0;
            for (int k = 0; k < blockSize; k++) {
                sum += A[i * blockSize + k] * B[k * blockSize + j];
            }
            C[i * blockSize + j] += sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBlocksRow = M / BLOCK_SIZE;
    int numBlocksCol = N / BLOCK_SIZE;
    int numBlocks = numBlocksRow * numBlocksCol;
    int blocksPerProc = numBlocks / size;

    double *A = NULL, *B = NULL, *C = NULL;
    double *localA = NULL, *localB = NULL, *localC = NULL;

    if (rank == 0) {
        A = (double*)malloc(M * K * sizeof(double));
        B = (double*)malloc(K * N * sizeof(double));
        C = (double*)malloc(M * N * sizeof(double));

        double A[8][8] = {
                {0.01900078, 0.08548636, -0.3314322, 0.02440604, -0.6461445, -0.7460251, 0.5706861, -0.06391164},
                {-0.3681220, 0.3359572, -0.8358069, 0.1730207, -0.7762620, -0.9918196, -0.1155304, -0.6060053},
                {0.4034051, -0.6577163, -0.4593707, 0.4775171, 0.7053617, -0.9989944, -0.4241922, -0.1056461},
                {-0.2655067, 0.6615583, -0.8514753, 0.7433776, 0.6223437, -0.6140962, 0.7067059, 0.4568796},
                {0.7526247, 0.7238108, 0.3524515, 0.1559346, 0.9097746, 0.1735603, 0.05194394, 0.9321860},
                {-0.2580107, 0.5384423, 0.6203171, 0.9294072, 0.8085664, -0.2500817, 0.4568334, 0.5136816},
                {0.05050480, -0.8948411, -0.07157754, -0.3329414, -0.7221695, -0.9464281, -0.4673714, -0.06788623},
                {-0.1443309, 0.9334593, 0.1383701, -0.2143441, 0.3723747, -0.07046941, 0.5669680, 0.7379329}
        };
    // Define matrix B
        double B[8][8] = {
                {-0.7319215, 0.8165651, -0.04029224, 0.8852518, 0.5361704, -0.2435351, -0.5578819, -0.3161686},
                {0.6921052, 0.5488294, -0.9377434, -0.2239507, 0.5038825, -0.1625671, 0.05086909, -0.08637625},
                {-0.9638247, -0.2678706, 0.2168604, 0.8610245, -0.8182718, -0.7231037, -0.01094956, 0.1762435},
                {0.8817096, 0.8885760, 0.5845223, 0.8227684, 0.7079334, -0.2383066, 0.7650440, 0.1702999},
                {0.3401642, 0.7696433, -0.6350064, -0.3245633, -0.6507111, -0.4004399, 0.5476602, -0.8601459},
                {0.5999431, -0.1096578, 0.05024259, 0.06562465, -0.9604593, -0.9579758, -0.8819950, 0.6378629},
                {-0.7135664, -0.6739431, 0.7975177, 0.4041545, -0.1618930, -0.5223948, -0.7603972, -0.1936786},
                {0.2209206, -0.1503654, -0.3824202, 0.9420213, -0.4072602, -0.9257813, 0.09560251, 0.1709661}
        };

        for (int i = 0; i < M * N; i++) C[i] = 0.0;
    }

    localA = (double*)malloc(blocksPerProc * BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    localB = (double*)malloc(K * N * sizeof(double));
    localC = (double*)malloc(blocksPerProc * BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // 分发A的块
    MPI_Scatter(A, blocksPerProc * BLOCK_SIZE * BLOCK_SIZE, MPI_DOUBLE, 
                localA, blocksPerProc * BLOCK_SIZE * BLOCK_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 广播整个B
    MPI_Bcast(B, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算本地块
    for (int b = 0; b < blocksPerProc; b++) {
        int blockRow = (rank * blocksPerProc + b) / numBlocksCol;
        int blockCol = (rank * blocksPerProc + b) % numBlocksCol;
        for (int k = 0; k < K / BLOCK_SIZE; k++) {
            blockMatrixMultiply(&localA[b * BLOCK_SIZE * BLOCK_SIZE],
                                &B[(k * BLOCK_SIZE * N) + (blockCol * BLOCK_SIZE)],
                                &localC[b * BLOCK_SIZE * BLOCK_SIZE],
                                BLOCK_SIZE);
        }
    }

    // 收集结果
    MPI_Gather(localC, blocksPerProc * BLOCK_SIZE * BLOCK_SIZE, MPI_DOUBLE,
               C, blocksPerProc * BLOCK_SIZE * BLOCK_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // 打印结果
        printf("Result matrix C:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f ", C[i * N + j]);
            }
            printf("\n");
        }

        free(A);
        free(B);
        free(C);
    }

    free(localA);
    free(localB);
    free(localC);

    MPI_Finalize();
    return 0;
}

