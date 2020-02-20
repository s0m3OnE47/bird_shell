# include <stdio.h>
# include<stdlib.h>
# define N 10000
# define BLOCK_SIZE 16
# define SHARED_MEM (BLOCK_SIZE*BLOCK_SIZE)



__global__ void MatMul(int *A, int *B,int*C, int MatDim) {

    __shared__ int tempA[SHARED_MEM];
    __shared__ int tempB[SHARED_MEM];

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*blockDim.y + threadIdx.y;
    int col = bx*blockDim.x + threadIdx.x;

    int temp =0;

    for(int i=0; i< (MatDim + BLOCK_SIZE - 1)/BLOCK_SIZE; ++i){

      if(row < MatDim  && (i*BLOCK_SIZE + ty < MatDim) ){
        tempA[ty*BLOCK_SIZE + tx] = A[i*MatDim*BLOCK_SIZE + ty*MatDim + row];
      }
      else
        tempA[ty*BLOCK_SIZE + tx] = 0;

      if(col < MatDim  && (i*BLOCK_SIZE + ty < MatDim)){
        tempB[ty*BLOCK_SIZE + tx] = B[i*MatDim*BLOCK_SIZE + ty*MatDim + col];
      }
      else
        tempB[ty*BLOCK_SIZE + tx] = 0;

      __syncthreads();

      for(int k=0;k<BLOCK_SIZE;++k){
        temp += tempA[ty*BLOCK_SIZE +k] * tempB[k*BLOCK_SIZE + tx];
      }
      __syncthreads();
    }

    if(row<MatDim && col<MatDim)
      C[row*MatDim + col] = temp;
}

int A[N][N], B[N][N],C[N][N], AT[N][N];

int main(){
  int *d_A, *d_B, *d_C;
  int size = N*N*sizeof(int);
  dim3 threads_per_block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 blocks_in_grid((N+BLOCK_SIZE-1)/BLOCK_SIZE,(N+BLOCK_SIZE-1)/BLOCK_SIZE);
  cudaEvent_t start, stop;
  float time;
 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i=0;i<N;++i)
    for(int j=0;j<N;++j)
      A[i][j] = 5;
    for(int i=0;i<N;++i)
    for(int j=0;j<N;++j)
      B[i][j] = 5;
    for(int i=0;i<N;++i)
    for(int j=0;j<N;++j)
      AT[i][j] = A[j][i];
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);
    cudaEventRecord( start, 0 );



    cudaMemcpy(d_A,AT,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    MatMul<<<blocks_in_grid,threads_per_block>>>(d_A,d_B,d_C,N);
    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    cudaEventElapsedTime( &time, start, stop );




    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  

  printf("Elapsed time is: %f\n",time);
return 0;

}
