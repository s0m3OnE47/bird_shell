#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void matrixMul (int *A_gpu,int *B_gpu,int *C_gpu,int N) {
	int k, accu = 0,i,j;
	i = blockIdx.y * blockDim.y + threadIdx.y;
	j = blockIdx.x * blockDim.x + threadIdx.x;

	if(i<N && j<N){
		for(k=0; k < N-1; k++) {
			accu += A_gpu[k*N+i] * B_gpu[k*N+j];
		}
	}

	C_gpu[i*N+j] = accu;
}

int main(int argc, char** argv) {
	int N;
	if(argc!=2) {
		printf("Invalid arguments");
		return -1;
	}
	
	N = atoi(argv[1]);

	int *A_cpu=(int *)malloc(N*N*sizeof(int)), *B_cpu=(int *)malloc(N*N*sizeof(int)) , *C_cpu=(int *)malloc(N*N*sizeof(int));
	int *A_gpu, *B_gpu, *C_gpu;
	int size= N*N*sizeof(int);
	int i,j,temp;
	cudaEvent_t start, stop;
	float time;

	for(i=0; i<N; i++){
		for(j=0;j<N;j++){
			*(A_cpu+i*N+j) = 1;
			*(B_cpu+i*N+j) = 2;
		}
	}

	for(i=0;i<N;i++){
		for(j=0;j<N;j++){
			temp = *(A_cpu+i*N+j);
			*(A_cpu+i*N+j) = *(A_cpu+j*N+i);
			*(A_cpu+j*N+i) = temp;
		}
	}

	cudaMalloc((void **)&A_gpu, size);
	cudaMalloc((void **)&B_gpu, size);
	cudaMalloc((void **)&C_gpu, size);

	A_cpu = (int *)malloc(size);
	B_cpu = (int *)malloc(size);
	C_cpu = (int *)malloc(size);	

	dim3 dimBlock(16,16);
	dim3 dimGrid((N+15)/16, (N+15)/16);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(A_gpu, A_cpu, size, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B_cpu, size, cudaMemcpyHostToDevice);

	cudaEventRecord(start,0);
	matrixMul<<<dimGrid, dimBlock>>>(A_gpu, B_gpu, C_gpu, N);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaMemcpy(C_cpu, C_gpu, size, cudaMemcpyDeviceToHost);

	free(A_cpu);
	free(B_cpu);
	free(C_cpu);
	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(C_gpu);

	printf("time for %d (excluding memcpy)= %f\n",N,time);
	return 0;
}

