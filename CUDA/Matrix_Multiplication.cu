/*
command for run 
nvcc -arch=sm_75 matrix.cu -o miju
time ./miju 10 50 & sleep 2
*/
//%%writefile matrix.cu
#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixMul(float* A, float* B, float* C, int M, int N, int P, int offset) {
    int k = threadIdx.x + offset;

    float* a = A + k * M * N;
    float* b = B + k * N * P;
    float* c = C + k * M * P;

    for(int i = 0; i < M; i++) {
        for(int j = 0; j < N; j++) {
            for(int l = 0; l < P; l++) {
                //c[i][l] += a[i][j] * b[j][l];
                c[i * P + l] = a[i * N + j] * b[j * P + l];
            }
        }
    }
}

int main(int argc, char *argv[]) {

    int T = atoi(argv[1]); //koyta thread use korte parbo
    int K = atoi(argv[2]); //koita matrix gun

    //100 gun, thread 10,

    int M = 400, N = 400, P = 400;

    int SizeA = M * N * K;
    int SizeB = N * P * K;
    int SizeC = M * P * K;

    //memory alocate (cpu allocate)
    float *h_A = new float[SizeA];
    float *h_B = new float[SizeB];
    float *h_C = new float[SizeC];


    //malloc (gpu allocate)
    float *d_A;
    cudaMalloc(&d_A, SizeA * sizeof(float));
    float *d_B;
    cudaMalloc(&d_B, SizeB * sizeof(float));
    float *d_C;
    cudaMalloc(&d_C, SizeC * sizeof(float));

    //data initialize
    for (int i = 0; i < SizeA; i++) {
        h_A[i] = rand();
    }
    for(int i = 0; i < SizeB; i++) {
        h_B[i] = rand();
    }


    //copy from host to device
    cudaMemcpy(d_A, h_A, SizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SizeB * sizeof(float), cudaMemcpyHostToDevice);

    //cuda process suru
    int gunKorteHobe = K;
    int offset = 0;
    while(gunKorteHobe > 0){

        int currentBatch = min(gunKorteHobe, T);

        matrixMul<<<1,currentBatch>>>(d_A, d_B, d_C, M, N, P, offset);
        cudaDeviceSynchronize();

        gunKorteHobe -= currentBatch;
        offset += currentBatch;
    }

    //let's copy back to cpu
    cudaMemcpy(h_C, d_C, SizeC * sizeof(float), cudaMemcpyDeviceToHost);

    for (int k = 0; k < K; k++) {
        cout << "Matrix C[" << k << "] (Result of multiplication " << k + 1 << "):" << endl;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < P; j++) {
                cout << h_C[k * M * P + i * P + j] << " ";
            }
            cout << endl;
        }
        cout << endl; // blank line between matrices
    }


    cout << "All operation done" << endl;

}
