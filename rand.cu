#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

__global__ void generate( curandState* globalState, float* array ) 
{
    int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState; 
    array[ind] = RANDOM;
}

int main( int argc, char** argv) 
{
    float   *host;
    float   *dev;  

    int i;
    int N = 3;

    dim3 tpb(N,1,1);
    curandState* devStates;
    cudaMalloc ( &devStates, N*sizeof( curandState ) );
    
    // setup seeds
    setup_kernel <<< 1, tpb >>> ( devStates, time(NULL) );

    host = (float*)malloc(sizeof(float) * N);
    cudaMalloc( (void**)&dev, sizeof(float) * N);

    for(i=0; i<N; i++)
    {
      host[i] = 0;
    }

    // generate random numbers
    generate <<< 1, tpb >>> ( devStates, dev );

    cudaMemcpy( host, dev, N*sizeof(float), cudaMemcpyDeviceToHost );

    for(i=0; i<N; i++)
    {
      printf("Random %d: %f\n", i, host[i]);
    }

    return 0;
}
