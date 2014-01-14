#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cuda.h>
#include <curand_kernel.h>

//#define DEBUG 1
//#define INFO 1

#define ERROR_ROULETTE 2

#define NUMBER_ORGANISMS_PER_BLOCK 40
#define NUMBER_BLOCK 100

#define MUTATION_RATE 0.001

#define TOTAL_GENERATIONS 500
#define TOTAL_CALLS 100

typedef struct {
  double x[8];
  double fitness;
  double p;
} Individuo;

__global__ void GA(curandState*, Individuo*, Individuo*);
__global__ void setup_rnd_kernel(curandState*, unsigned long);
__device__ float rand(curandState*);
__device__ void EvaluateOrganisms(Individuo*);
__device__ void ProduceNextGeneration(curandState*, Individuo*, Individuo*);
__device__ int SelectOneOrganism(curandState*, Individuo*);
__device__ double atomicAdd(double*, double);
__host__ int totalDeRestricoesQuebradas(Individuo);
__host__ void exibirRestricoesQuebradas(Individuo);
__host__ void exibirErroOtimo(Individuo);
__host__ void cudasafe(cudaError_t, char*);
__host__ void InitializeOrganisms(Individuo*);

void cudasafe( cudaError_t err, char* str)
{
 if (err != cudaSuccess)
 {
  printf("%s failed with error code %i\n",str,err);
  exit(1);
 }
}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

/**
 * Funcao de avaliacao utilizada:
 * min G(x) = x1 + x2 + x3
 * s.t. 
 * 1 - 0.0025(x4 + x6) >= 0,
 * 1 - 0.0025(x5 + x7 - x4) >= 0,
 * 1 - 0.01(x8 - x5) >= 0,
 * x1*x6 - 833.33252*x4 - 100*x1 + 83333.333 >= 0,
 * x2*x7 - 1250*x5 - x2*x4 + 1250*x4 >= 0,
 * x3*x8 - 1250000 - x3*x5 + 2500*x5 >= 0,
 * 100 <= x1 <= 10000,
 * 1000 <= xi <= 10000, i = 2, 3,
 * 10 <= xi <= 1000, i = 4, ..., 8.
 */
__device__ void EvaluateOrganisms(Individuo* currentGeneration)
{
  __shared__ double totalOfFitnesses;
  __shared__ double totalP;

  int  organism;
  double currentFitness;
  double constraint;

  if (threadIdx.x == 0)
  {
    totalOfFitnesses = 0;
    totalP = 0;
  }
  __syncthreads();

  organism = threadIdx.x;

  currentFitness = 0;
  
  // Calculando a função de avaliação
  currentFitness += currentGeneration[organism].x[0] + currentGeneration[organism].x[1] + currentGeneration[organism].x[2];
  
  // Analisando as restricoes e adicionando as punicoes
  //1 - 0.0025(x4 + x6) >= 0
  constraint = 1 - 0.0025*(currentGeneration[organism].x[3] + currentGeneration[organism].x[5]);
  currentFitness += fabs(min(0.0, constraint));
  
  //1 - 0.0025(x5 + x7 - x4) >= 0
  constraint = 1 - 0.0025*(currentGeneration[organism].x[4] + currentGeneration[organism].x[6] - currentGeneration[organism].x[3]);
  currentFitness += fabs(min(0.0, constraint));
  
  //1 - 0.01(x8 - x5) >= 0
  constraint = 1 - 0.01*(currentGeneration[organism].x[7] - currentGeneration[organism].x[4]);
  currentFitness += fabs(min(0.0, constraint));
  
  //x1*x6 - 833.33252*x4 - 100*x1 + 83333.333 >= 0
  constraint = currentGeneration[organism].x[0]*currentGeneration[organism].x[5] - 833.33252*currentGeneration[organism].x[3] - 100*currentGeneration[organism].x[0] + 83333.333;
  currentFitness += fabs(min(0.0, constraint));
  
  //x2*x7 - 1250*x5 - x2*x4 + 1250*x4 >= 0
  constraint = currentGeneration[organism].x[1]*currentGeneration[organism].x[6] - 1250*currentGeneration[organism].x[4] - currentGeneration[organism].x[1]*currentGeneration[organism].x[3] + 1250*currentGeneration[organism].x[3];
  currentFitness += fabs(min(0.0, constraint));
  
  //x3*x8 - 1250000 - x3*x5 + 2500*x5 >= 0
  constraint = currentGeneration[organism].x[2]*currentGeneration[organism].x[7] - 1250000 - currentGeneration[organism].x[2]*currentGeneration[organism].x[4] + 2500*currentGeneration[organism].x[4];
  currentFitness += fabs(min(0.0, constraint));
  
  //100 <= x1
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[0] - 100));
  
  //x1 <= 10000
  currentFitness += fabs(min(0.0, 10000 - currentGeneration[organism].x[0]));
  
  //1000 <= xi <= 10000, i = 2, 3,
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[1] - 1000));
  currentFitness += fabs(min(0.0, 10000 - currentGeneration[organism].x[1]));
  
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[2] - 1000));
  currentFitness += fabs(min(0.0, 10000 - currentGeneration[organism].x[2]));
  
  //10 <= xi <= 1000, i = 4, ..., 8.
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[3] - 10));
  currentFitness += fabs(min(0.0, 1000 - currentGeneration[organism].x[3]));
  
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[4] - 10));
  currentFitness += fabs(min(0.0, 1000 - currentGeneration[organism].x[4]));
  
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[5] - 10));
  currentFitness += fabs(min(0.0, 1000 - currentGeneration[organism].x[5]));
  
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[6] - 10));
  currentFitness += fabs(min(0.0, 1000 - currentGeneration[organism].x[6]));
  
  currentFitness += fabs(min(0.0, currentGeneration[organism].x[7] - 10));
  currentFitness += fabs(min(0.0, 1000 - currentGeneration[organism].x[7]));
  
  currentGeneration[organism].fitness = currentFitness;

  atomicAdd(&totalOfFitnesses, currentFitness);
  __syncthreads();

  // Ajustando a probabilidade de ser escolhido para gerar a proxima geracao
  currentGeneration[organism].p = 1 - (currentGeneration[organism].fitness / totalOfFitnesses);

  atomicAdd(&totalP, currentGeneration[organism].p);
  __syncthreads();
  
  currentGeneration[organism].p = currentGeneration[organism].p / totalP;
}

__device__ int SelectOneOrganism(curandState* localState, Individuo* currentGeneration)
{
  int organism;
  double runningTotal;
  double randomSelectPoint;

  runningTotal = 0;
  randomSelectPoint = rand(localState);

  for(organism=0; organism<NUMBER_ORGANISMS_PER_BLOCK; organism++)
  {
    runningTotal += currentGeneration[organism].p;
    if(runningTotal >= randomSelectPoint)
    {
      return organism;
    }
  }
  
  return -1;
}

/**
 * Crossover baseado no line-crossover
 * Mutation baseado em fine-mutation
 */
__device__ void ProduceNextGeneration(curandState* localState, Individuo* currentGeneration, Individuo* nextGeneration)
{
  int organism;
  int gene;
  int parentOne;
  int parentTwo;
  
  double mutate;
  double lambda;
  
  double bestFitness;
  int bestOrganism;

  #ifdef DEBUG
    printf("[ProduceNextGeneration] J: %d\n", j);
  #endif

  organism = threadIdx.x;

  if (organism == 0)
  {
    // 1 Elite sendo carregados para proxima geracao
    bestFitness = currentGeneration[0].fitness;
    bestOrganism = 0;
    
    for(organism=0; organism<NUMBER_ORGANISMS_PER_BLOCK; organism++)
    {
      if (currentGeneration[organism].fitness < bestFitness)
      {
        bestFitness = currentGeneration[organism].fitness;
        bestOrganism = organism;
      }
    }

    for(gene=0; gene<8; gene++)
    {
      nextGeneration[0].x[gene] = currentGeneration[bestOrganism].x[gene];
    }
  }
  else
  {
    parentOne = SelectOneOrganism(localState, currentGeneration);
    parentTwo = SelectOneOrganism(localState, currentGeneration);
    
    lambda = rand(localState);
    
    for(gene=0; gene<8; gene++)
    {
      nextGeneration[organism].x[gene] = lambda*currentGeneration[parentOne].x[gene] + (1 - lambda)*currentGeneration[parentTwo].x[gene];
      
      //mutacao
      mutate = rand(localState);
      if (mutate <= MUTATION_RATE)
      {
        nextGeneration[organism].x[gene] = nextGeneration[organism].x[gene] + rand(localState);
      }
    }
  }
}

__global__ void setup_rnd_kernel ( curandState* state, unsigned long seed )
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
} 

__device__ float rand( curandState* localRand ) 
{
    int ind = threadIdx.x;
    curandState localState = localRand[ind];
    float rnd = curand_uniform( &localState );
    localRand[ind] = localState;
    
    return rnd;
}

__global__ void GA(curandState* globalRand, Individuo* deviceBestGeneration)
{
  __shared__ int generations;
  __shared__ Individuo currentGeneration[NUMBER_ORGANISMS_PER_BLOCK];
  __shared__ Individuo nextGeneration[NUMBER_ORGANISMS_PER_BLOCK];
  __shared__ curandState localRand[NUMBER_ORGANISMS_PER_BLOCK];
  
  int bloco, gene;
  int posLocal = threadIdx.x;
  int posGlobal = blockIdx.x;
  
  localRand[posLocal] = globalRand[posGlobal];

  //copiar do melhor individuo da ultima chamada
  if (threadIdx.x == 1)
  {
    for (gene=0; gene<8; gene++)
    {
      bloco = (posGlobal + 1) % NUMBER_BLOCK;
      currentGeneration[posLocal].x[gene] = deviceBestGeneration[bloco].x[gene];
    }
    generations = 1;
  }
  else 
  {
    for (gene=0; gene<8; gene++)
    {
      currentGeneration[posLocal].x[gene] = rand(localRand) * 10000;
    }
  }
  currentGeneration[posLocal].fitness = 0;
  
  __syncthreads();

  while(generations < TOTAL_GENERATIONS)
  {
    EvaluateOrganisms(currentGeneration);
    __syncthreads();    
    
    ProduceNextGeneration(localRand, currentGeneration, nextGeneration);
    __syncthreads();

    //trocar os ponteiros
    nextGeneration[posLocal] = currentGeneration[posLocal];

    if (threadIdx.x == 0)
    {
      generations++;
    }

    __syncthreads();
  }
  
  /* Copiando o melhor de volta para memória global */
  if (threadIdx.x == 0)
  {
    for(gene=0; gene<8; gene++)
    {
      deviceBestGeneration[posGlobal].x[gene] = nextGeneration[0].x[gene];
    }
    
    deviceBestGeneration[posGlobal].fitness = nextGeneration[0].fitness;
  }
  
  __syncthreads();
}

__host__ int totalDeRestricoesQuebradas(Individuo individuo)
{
  int restricoesQuebradas = 0;
  int gene;
  double constraint;
  
  constraint = 1 - 0.0025*(individuo.x[3] + individuo.x[5]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = 1 - 0.0025*(individuo.x[4] + individuo.x[6] - individuo.x[3]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = 1 - 0.01*(individuo.x[7] - individuo.x[4]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = individuo.x[0]*individuo.x[5] - 833.33252*individuo.x[3] - 100*individuo.x[0] + 83333.333;
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = individuo.x[1]*individuo.x[6] - 1250*individuo.x[4] - individuo.x[1]*individuo.x[3] + 1250*individuo.x[3];
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = individuo.x[2]*individuo.x[7] - 1250000 - individuo.x[2]*individuo.x[4] + 2500*individuo.x[4];
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0.0, individuo.x[0] - 100);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0.0, 10000 - individuo.x[0]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0.0, individuo.x[1] - 1000);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0.0, 10000 - individuo.x[1]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0.0, individuo.x[2] - 1000);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0.0, 10000 - individuo.x[2]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  for(gene=3; gene<8; gene++)
  {
    constraint = min(0.0, individuo.x[gene] - 10);
    if (constraint < 0)
    {
      restricoesQuebradas++;
    }
  
    constraint = min(0.0, 1000 - individuo.x[gene]);
    if (constraint < 0)
    {
      restricoesQuebradas++;
    }
  }
  
  return restricoesQuebradas;
}

__host__ void exibirRestricoesQuebradas(Individuo individuo)
{
  int gene;
  double constraint;
  
  constraint = 1 - 0.0025*(individuo.x[3] + individuo.x[5]);
  if (constraint < 0)
  {
    printf("Restricao 1 - 0.0025(x4 + x6) >= 0 não respeitada.\n");
  }
  
  constraint = 1 - 0.0025*(individuo.x[4] + individuo.x[6] - individuo.x[3]);
  if (constraint < 0)
  {
    printf("Restricao 1 - 0.0025(x5 + x7 - x4) >= 0 não respeitada.\n");
  }
  
  constraint = 1 - 0.01*(individuo.x[7] - individuo.x[4]);
  if (constraint < 0)
  {
    printf("Restricao 1 - 0.01(x8 - x5) >= 0 não respeitada.\n");
  }
  
  constraint = individuo.x[0]*individuo.x[5] - 833.33252*individuo.x[3] - 100*individuo.x[0] + 83333.333;
  if (constraint < 0)
  {
    printf("Restricao x1*x6 - 833.33252*x4 - 100*x1 + 83333.333 >= 0 não respeitada.\n");
  }
  
  constraint = individuo.x[1]*individuo.x[6] - 1250*individuo.x[4] - individuo.x[1]*individuo.x[3] + 1250*individuo.x[3];
  if (constraint < 0)
  {
    printf("Restricao x2*x7 - 1250*x5 - x2*x4 + 1250*x4 >= 0 não respeitada.\n");
  }
  
  constraint = individuo.x[2]*individuo.x[7] - 1250000 - individuo.x[2]*individuo.x[4] + 2500*individuo.x[4];
  if (constraint < 0)
  {
    printf("Restricao x3*x8 - 1250000 - x3*x5 + 2500*x5 >= 0 não respeitada.\n");
  }
  
  constraint = min(0.0, individuo.x[0] - 100);
  if (constraint < 0)
  {
    printf("Restricao 100 <= x1 não respeitada.\n");
  }
  
  constraint = min(0.0, 10000 - individuo.x[0]);
  if (constraint < 0)
  {
    printf("Restricao x1 <= 10000 não respeitada.\n");
  }
  
  constraint = min(0.0, individuo.x[1] - 1000);
  if (constraint < 0)
  {
    printf("Restricao 1000 <= x2 não respeitada.\n");
  }
  
  constraint = min(0.0, 10000 - individuo.x[1]);
  if (constraint < 0)
  {
    printf("Restricao x2 <= 10000 não respeitada.\n");
  }
  
  constraint = min(0.0, individuo.x[2] - 1000);
  if (constraint < 0)
  {
    printf("Restricao 1000 <= x3 não respeitada.\n");
  }
  
  constraint = min(0.0, 10000 - individuo.x[2]);
  if (constraint < 0)
  {
    printf("Restricao x3 <= 10000 não respeitada.\n");
  }
  
  for(gene=3; gene<8; gene++)
  {
    constraint = min(0.0, individuo.x[gene] - 10);
    if (constraint < 0)
    {
      printf("Restricao 10 <= x%d não respeitada.\n", gene+1);
    }
  
    constraint = min(0.0, 1000 - individuo.x[gene]);
    if (constraint < 0)
    {
      printf("Restricao x%d <= 1000 não respeitada.\n", gene+1);
    }
  }
}

__host__ void exibirErroOtimo(Individuo individuo)
{
  double objetivo;
  
  objetivo = individuo.x[0] + individuo.x[1] + individuo.x[2];
  
  printf("Objetivo %f. Otimo %f. Erro %f\n", objetivo, 7049.330923, objetivo - 7049.330923);
  printf("X1 %f. Otimo %f. Erro %f\n", individuo.x[0], 579.3167, individuo.x[0] - 579.3167);
  printf("X2 %f. Otimo %f. Erro %f\n", individuo.x[1], 1359.943, individuo.x[1] - 1359.943);
  printf("X3 %f. Otimo %f. Erro %f\n", individuo.x[2], 5110.071, individuo.x[2] - 5110.071);
  printf("X4 %f. Otimo %f. Erro %f\n", individuo.x[3], 182.0174, individuo.x[3] - 182.0174);
  printf("X5 %f. Otimo %f. Erro %f\n", individuo.x[4], 295.5985, individuo.x[4] - 295.5985);
  printf("X6 %f. Otimo %f. Erro %f\n", individuo.x[5], 217.9799, individuo.x[5] - 217.9799);
  printf("X7 %f. Otimo %f. Erro %f\n", individuo.x[6], 286.4162, individuo.x[6] - 286.4162);
  printf("X8 %f. Otimo %f. Erro %f\n", individuo.x[7], 395.5979, individuo.x[7] - 395.5979);
}

__host__ void InitializeOrganisms(Individuo *hostBestGeneration)
{
  int organism;
  int gene;

  for(organism=0; organism<NUMBER_BLOCK; organism++)
  {
    for (gene=0; gene<8; gene++)
    {
      hostBestGeneration[organism].x[gene] = rand() % 10000;
    }
    
    hostBestGeneration[organism].fitness = 0;
  }
}

int main(void)
{
  Individuo *hostFirstGeneration, *hostBestGeneration;
  Individuo *deviceBestGeneration;

  int organism, gene;
  int call;
  int restricoesQuebradas, menorNumeroRestricoesQuebradas;
  int melhorIndividuo;
  struct timeval inicio, fim;
  double tempo;
  
  curandState* devStates;
  
  cudaSetDevice(0);

  // Setup execution parameters
  dim3 threads(NUMBER_ORGANISMS_PER_BLOCK, 1, 1);
  dim3 grid(NUMBER_BLOCK, 1, 1);

  /* initialize random seed: */
  srand (time(NULL));

  cudasafe(cudaHostAlloc((void **)&hostFirstGeneration, sizeof(Individuo) * NUMBER_BLOCK, 0), "cudaHostAlloc\0");
  cudasafe(cudaHostAlloc((void **)&hostBestGeneration, sizeof(Individuo) * NUMBER_BLOCK, 0), "cudaHostAlloc\0");
  InitializeOrganisms(hostFirstGeneration);
  
  cudasafe(cudaMalloc ((void **) &deviceBestGeneration, sizeof(Individuo) * NUMBER_BLOCK), "cudaMalloc\0");
  
  gettimeofday(&inicio, NULL); 
  
  cudasafe(cudaMalloc ( &devStates, NUMBER_BLOCK * NUMBER_ORGANISMS_PER_BLOCK*sizeof( curandState ) ), "cudaMalloc\0");
  setup_rnd_kernel <<< grid, threads >>> ( devStates, time(NULL) );

  printf("Total de memória alocada: %lu\n", sizeof(Individuo) * NUMBER_BLOCK);
  printf("Copiando dados para a GPU...\n");
  cudasafe(cudaMemcpy(deviceBestGeneration, hostFirstGeneration, sizeof(Individuo) * NUMBER_BLOCK, cudaMemcpyHostToDevice), "cudaMemcpyHostToDevice\0");
  printf("Dados copiados para a GPU. Iniciando kernel.\n");
  for(call=0; call<TOTAL_CALLS; call++)
  {
    GA<<< grid, threads >>>(devStates, deviceBestGeneration);
    cudaDeviceSynchronize();
    printf("Chamada %d acabou\n", call);
  }
  printf("Copiando dados de volta para a CPU...\n");
  cudasafe(cudaMemcpy(hostBestGeneration, deviceBestGeneration, sizeof(Individuo) * NUMBER_BLOCK, cudaMemcpyDeviceToHost), "cudaMemcpyDeviceToHost\0");
  printf("Dados copiados para a CPU.\n");
 
  gettimeofday(&fim, NULL);
  tempo = (fim.tv_sec - inicio.tv_sec)*1000 + (fim.tv_usec - inicio.tv_usec)/1000; //calcula tempo em milisegundos
  printf("Tempo por geração: %.3lf(ms)\n", tempo/(TOTAL_GENERATIONS*TOTAL_CALLS));
  printf("Tempo total: %.3lf(ms)\n", tempo);
 
  melhorIndividuo = 0;
  menorNumeroRestricoesQuebradas = totalDeRestricoesQuebradas(hostBestGeneration[0]);
  
  for(organism=0; organism<NUMBER_BLOCK; organism++)
  {
    restricoesQuebradas = totalDeRestricoesQuebradas(hostBestGeneration[organism]);
    if (restricoesQuebradas < menorNumeroRestricoesQuebradas)
    {
      menorNumeroRestricoesQuebradas = restricoesQuebradas;
      melhorIndividuo = organism;
    }
  }

  printf("Melhor fitness: %.2f\n", hostBestGeneration[melhorIndividuo].fitness);

  for(gene=0; gene<8; gene++)
  {
    printf("X%d: %.2f\n", gene+1, hostBestGeneration[melhorIndividuo].x[gene]);
  }
  
  printf("Restricoes quebradas:\n");
  exibirRestricoesQuebradas(hostBestGeneration[melhorIndividuo]);
  printf("Total: %d\n", menorNumeroRestricoesQuebradas);
  
  exibirErroOtimo(hostBestGeneration[melhorIndividuo]);

  cudasafe(cudaFreeHost(hostBestGeneration), "cudaFreeHost\0");
  cudasafe(cudaFree(devStates), "cudaFree\0");
  cudasafe(cudaFree(deviceBestGeneration), "cudaFree\0");

  return 0;
}
