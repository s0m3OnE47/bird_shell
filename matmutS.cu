# include <stdio.h>
# include<stdlib.h>
# include<assert.h>
# define N 100
# define LOOP 5
# define BLOCK_SIZE 16
# define SHARED_MEM (BLOCK_SIZE*BLOCK_SIZE)

void test_results(int A[N][N], int B[N][N],int C[N][N]){
  int temp;
  for(int i=0;i<N;++i){
    for(int j=0;j<N;++j){
      temp =0;
      for(int k=0; k<N; ++k){
        temp += A[i][k]*B[k][j];
      }
      assert(temp == C[i][j]);
    }
  }
}


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

      if(row < MatDim  && (i*BLOCK_SIZE + tx < MatDim) ){
        tempA[ty*BLOCK_SIZE + tx] = A[row*MatDim + i*BLOCK_SIZE + tx];
      }
      else
        tempA[ty*BLOCK_SIZE + tx] = 0;

      if(col < MatDim  && i*BLOCK_SIZE + ty < MatDim){
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

int A[N][N], B[N][N],C[N][N];

int main(){
  int *d_A, *d_B, *d_C;
  int size = N*N*sizeof(int);
  dim3 threads_per_block(BLOCK_SIZE,BLOCK_SIZE);
  dim3 blocks_in_grid((N+BLOCK_SIZE-1)/BLOCK_SIZE,(N+BLOCK_SIZE-1)/BLOCK_SIZE);
  cudaEvent_t start, stop;
  float time[LOOP];
  for(int k=0;k<LOOP;++k){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i=0;i<N;++i)
    for(int j=0;j<N;++j)
      A[i][j] = rand()%N;
    for(int i=0;i<N;++i)
    for(int j=0;j<N;++j)
      B[i][j] = rand()%N;
    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);
    cudaEventRecord( start, 0 );

    cudaEventRecord( start, 0 );

    cudaMemcpy(d_A,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,size,cudaMemcpyHostToDevice);

    MatMul<<<blocks_in_grid,threads_per_block>>>(d_A,d_B,d_C,N);

    cudaMemcpy(C,d_C,size,cudaMemcpyDeviceToHost);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );

    cudaEventElapsedTime( &time[k], start, stop );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
  }
  /*for(int i=0;i<N;++i){
    for(int j=0;j<N;++j)
      printf("%d ",A[i][j]);
    printf("\n");
  }
  printf("\n");
  for(int i=0;i<N;++i){
    for(int j=0;j<N;++j)
      printf("%d ",B[i][j]);
    printf("\n");
  }
  printf("\n");
  for(int i=0;i<N;++i){
    for(int j=0;j<N;++j)
      printf("%d ",C[i][j]);
    printf("\n");
  }*/

test_results(A,B,C);
printf("Results verified\n");

float average =0;
for(int i=0;i<LOOP;++i)
  average = average + time[i];
average = average/LOOP;
  printf("Elapsed time is: %f\n",average);
return 0;

}
