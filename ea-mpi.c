#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <pthread.h>
#include <mpi.h>

//#define DEBUG 1
#define INFO 1

#define NTHREADS 2

#define ERROR_ROULETTE 2

#define NUMBER_ORGANISMS 400

#define MUTATION_RATE 0.001

#define TOTAL_GENERATIONS 500
#define TOTAL_CALLS 100

typedef struct {
  double x[8];
  double fitness;
  double p;
} Individuo;

pthread_t tid[NTHREADS];
pthread_barrier_t barreira;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

Individuo *currentGeneration, *nextGeneration, *bestGeneration;
int generations;
int rank, nprocs;
double totalOfFitnesses = 0;
double totalP = 0;

void AllocateMemory(void);
void Run(void);
void InitializeOrganisms(void);
void EvaluateOrganisms(int);
void ProduceNextGeneration(int);
int SelectOneOrganism(int);
double rand_normal(double, double);
void swapGenerations(Individuo**, Individuo**);
int min(int, int);
int totalDeRestricoesQuebradas(Individuo);
void exibirRestricoesQuebradas(Individuo);
void exibirErroOtimo(Individuo);

void AllocateMemory(void)
{
  currentGeneration = (Individuo*)malloc(sizeof(Individuo) * NUMBER_ORGANISMS);
  nextGeneration = (Individuo*)malloc(sizeof(Individuo) * NUMBER_ORGANISMS);
} 

void InitializeOrganisms(void)
{
  int organism;
  int i;

  // initialize the normal organisms
  for(organism=0; organism<NUMBER_ORGANISMS; organism++)
  {
    for (i=0; i<8; i++)
    {
      currentGeneration[organism].x[i] = rand() % 10000;
    }
    
    currentGeneration[organism].fitness = 0;
  }
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
void EvaluateOrganisms(int tid)
{
  int organism;
  double currentFitness;
  double localTotalOfFitnesses, localTotalP;
  double constraint;

  localTotalOfFitnesses = 0;

  for(organism=tid; organism<NUMBER_ORGANISMS; organism+=NTHREADS)
  {
    currentFitness = 0;
    
    // Calculando a função de avaliação
    currentFitness += currentGeneration[organism].x[0] + currentGeneration[organism].x[1] + currentGeneration[organism].x[2];
    
    // Analisando as restricoes e adicionando as punicoes
    //1 - 0.0025(x4 + x6) >= 0
    constraint = 1 - 0.0025*(currentGeneration[organism].x[3] + currentGeneration[organism].x[5]);
    currentFitness += fabs(min(0, constraint));
    
    //1 - 0.0025(x5 + x7 - x4) >= 0
    constraint = 1 - 0.0025*(currentGeneration[organism].x[4] + currentGeneration[organism].x[6] - currentGeneration[organism].x[3]);
    currentFitness += fabs(min(0, constraint));
    
    //1 - 0.01(x8 - x5) >= 0
    constraint = 1 - 0.01*(currentGeneration[organism].x[7] - currentGeneration[organism].x[4]);
    currentFitness += fabs(min(0, constraint));
    
    //x1*x6 - 833.33252*x4 - 100*x1 + 83333.333 >= 0
    constraint = currentGeneration[organism].x[0]*currentGeneration[organism].x[5] - 833.33252*currentGeneration[organism].x[3] - 100*currentGeneration[organism].x[0] + 83333.333;
    currentFitness += fabs(min(0, constraint));
    
    //x2*x7 - 1250*x5 - x2*x4 + 1250*x4 >= 0
    constraint = currentGeneration[organism].x[1]*currentGeneration[organism].x[6] - 1250*currentGeneration[organism].x[4] - currentGeneration[organism].x[1]*currentGeneration[organism].x[3] + 1250*currentGeneration[organism].x[3];
    currentFitness += fabs(min(0, constraint));
    
    //x3*x8 - 1250000 - x3*x5 + 2500*x5 >= 0
    constraint = currentGeneration[organism].x[2]*currentGeneration[organism].x[7] - 1250000 - currentGeneration[organism].x[2]*currentGeneration[organism].x[4] + 2500*currentGeneration[organism].x[4];
    currentFitness += fabs(min(0, constraint));
    
    //100 <= x1
    currentFitness += fabs(min(0, currentGeneration[organism].x[0] - 100));
    
    //x1 <= 10000
    currentFitness += fabs(min(0, 10000 - currentGeneration[organism].x[0]));
    
    //1000 <= xi <= 10000, i = 2, 3,
    currentFitness += fabs(min(0, currentGeneration[organism].x[1] - 1000));
    currentFitness += fabs(min(0, 10000 - currentGeneration[organism].x[1]));
    
    currentFitness += fabs(min(0, currentGeneration[organism].x[2] - 1000));
    currentFitness += fabs(min(0, 10000 - currentGeneration[organism].x[2]));
    
    //10 <= xi <= 1000, i = 4, ..., 8.
    currentFitness += fabs(min(0, currentGeneration[organism].x[3] - 10));
    currentFitness += fabs(min(0, 1000 - currentGeneration[organism].x[3]));
    
    currentFitness += fabs(min(0, currentGeneration[organism].x[4] - 10));
    currentFitness += fabs(min(0, 1000 - currentGeneration[organism].x[4]));
    
    currentFitness += fabs(min(0, currentGeneration[organism].x[5] - 10));
    currentFitness += fabs(min(0, 1000 - currentGeneration[organism].x[5]));
    
    currentFitness += fabs(min(0, currentGeneration[organism].x[6] - 10));
    currentFitness += fabs(min(0, 1000 - currentGeneration[organism].x[6]));
    
    currentFitness += fabs(min(0, currentGeneration[organism].x[7] - 10));
    currentFitness += fabs(min(0, 1000 - currentGeneration[organism].x[7]));
    
    currentGeneration[organism].fitness = currentFitness;
    localTotalOfFitnesses += currentFitness;
    
    if (currentFitness < 0)
    {
      printf("[Erro] Fitness: %.2f\n", currentFitness);
      printf("[Erro] X1: %.2f -> %.2f\n", currentGeneration[organism].x[0], fabs(min(0, currentGeneration[organism].x[0] - 100)));
      printf("[Erro] X2: %.2f -> %.2f\n", currentGeneration[organism].x[1], fabs(min(0, currentGeneration[organism].x[1] - 1000)));
      printf("[Erro] X3: %.2f -> %.2f\n", currentGeneration[organism].x[2], fabs(min(0, currentGeneration[organism].x[2] - 1000)));
    }
  }
  
  pthread_mutex_lock(&mutex);
    totalOfFitnesses += localTotalOfFitnesses;
  pthread_mutex_unlock(&mutex);
  
  pthread_barrier_wait(&barreira);

  localTotalP = 0;
  // Ajustando a probabilidade de ser escolhido para gerar a proxima geracao
  for(organism=tid; organism<NUMBER_ORGANISMS; organism+=NTHREADS)
  {
    currentGeneration[organism].p = 1 - (currentGeneration[organism].fitness / totalOfFitnesses);
    localTotalP += currentGeneration[organism].p;
  }

  pthread_mutex_lock(&mutex);
    totalP += localTotalP;
  pthread_mutex_unlock(&mutex);

  pthread_barrier_wait(&barreira);

  // Normalizando a probabilidade de ser escolhido para gerar a proxima geracao
  for(organism=tid; organism<NUMBER_ORGANISMS; organism+=NTHREADS)
  {
    currentGeneration[organism].p = currentGeneration[organism].p / totalP;
  }
  
  #ifdef DEBUG
    printf("[Evaluate] Total Fitness: %.2f\n", totalOfFitnesses);
    for(organism=tid; organism<NUMBER_ORGANISMS; organism+=NTHREADS)
    {
      printf("[Evaluate %d] P: %.8f, fitness: %.2f\n", organism, currentGeneration[organism].p, currentGeneration[organism].fitness);
    }
  #endif
}

int SelectOneOrganism(int tid)
{
  int organism;
  double runningTotal;
  double randomSelectPoint;

  runningTotal = 0;
  randomSelectPoint = ((double)rand()/(double)RAND_MAX);

  for(organism=0; organism<NUMBER_ORGANISMS; organism++)
  {
    runningTotal += currentGeneration[organism].p;
    if(runningTotal >= randomSelectPoint)
    {
      #ifdef DEBUG
        printf("[SelectOneOrganism%d] RandomSelect: %.2f\n", tid, randomSelectPoint);
        printf("[SelectOneOrganism%d] Individuo %d\n", tid, organism);
      #endif
      return organism;
    }
  }

  printf("ERRO FATAL NA ROLETA!\n");
  exit(ERROR_ROULETTE);
}

/**
 * Crossover baseado no line-crossover
 * Mutation baseado em fine-mutation
 */
void ProduceNextGeneration(int tid)
{
  int organism;
  int gene;
  int parentOne;
  int parentTwo;
  
  double mutate;
  double lambda;
  
  double bestFitness;
  int bestOrganism;

  // Apenas a thread 0 separa o elite
  if (tid == 0)
  {
    // 1 Elite sendo carregados para proxima geracao
    bestFitness = currentGeneration[0].fitness;
    bestOrganism = 0;
    
    for(organism=0; organism<NUMBER_ORGANISMS; organism++)
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

  // fill the nextGeneration data structure with the
  // children
  for(organism=tid; organism<NUMBER_ORGANISMS; organism+=NTHREADS)
  {
    if (organism == 0) continue;

    parentOne = SelectOneOrganism(tid);
    parentTwo = SelectOneOrganism(tid);
    
    lambda = ((double)rand()/(double)RAND_MAX);
    
    for(gene=0; gene<8; gene++)
    {
      nextGeneration[organism].x[gene] = lambda*currentGeneration[parentOne].x[gene] + (1 - lambda)*currentGeneration[parentTwo].x[gene];
      #ifdef DEBUG
        printf("[ProduceNextGeneration] Gene[%d]: %.2f\n", gene, nextGeneration[organism].x[gene]);
      #endif
      
      //mutacao
      mutate = ((double)rand()/(double)RAND_MAX);
      #ifdef DEBUG
        printf("[ProduceNextGeneration] Mutacao: %.4f\n", mutate);
      #endif
      if (mutate <= MUTATION_RATE)
      {
        nextGeneration[organism].x[gene] = nextGeneration[organism].x[gene] + rand_normal(0, fabs(currentGeneration[parentOne].x[gene] - currentGeneration[parentTwo].x[gene]));
        #ifdef DEBUG
          printf("[ProduceNextGeneration] Mutacao: %.2f\n", nextGeneration[organism].x[gene]);
        #endif
      }
    }
  }
}

void swapGenerations(Individuo** current, Individuo** next)
{
  Individuo *temp = *current;
  *current = *next;
  *next = temp;
}

/**
 * Box-Muller transform
 */
double rand_normal(double mean, double stddev)
{
  double x, y, r, d, n1;
  
  do
  {
    x = 2.0*rand()/RAND_MAX - 1;
    y = 2.0*rand()/RAND_MAX - 1;
    r = x*x + y*y;
  } while (r == 0.0 || r > 1.0);
  
  d = sqrt(-2.0*log(r)/r);
  n1 = x*d;
  
  return n1*stddev + mean;
}

/**
 * http://graphics.stanford.edu/~seander/bithacks.html#IntegerMinOrMax
 */
int min(int x, int y)
{
  return y ^ ((x ^ y) & -(x < y));
}

void *ThreadRun (void *threadid)
{
  int organism;
  double averageFitness, p;
  
  int tid = (int) threadid;

  #ifdef DEBUG
    printf("\tThread %d iniciou!\n", tid);
  #endif
  
  while(generations < TOTAL_GENERATIONS)
  {
    EvaluateOrganisms(tid);
    pthread_barrier_wait(&barreira);

    ProduceNextGeneration(tid);
    pthread_barrier_wait(&barreira);
    
    // apenas um
    if (tid == 0)
    {
      //trocar os ponteiros
      swapGenerations(&currentGeneration, &nextGeneration);

      totalOfFitnesses = 0;
      totalP = 0;
      
      /*
      averageFitness = 0;
      for(organism=0; organism<NUMBER_ORGANISMS; organism++)
      {
        averageFitness += currentGeneration[organism].fitness;
      }
      
      averageFitness /= NUMBER_ORGANISMS;
      
      #ifdef INFO
        printf("Média da geração %d: %.2f\n", generations, averageFitness);
      #endif
      */
      
      generations++;
      //printf("Geração %d acabou\n", generations);
    }

    pthread_barrier_wait(&barreira);
  }

  #ifdef DEBUG
    printf("\tThread %d terminou!\n", tid);
  #endif
  
  pthread_exit(NULL);
}

void Run(void)
{
  int t;
  int calls;

  pthread_barrier_init(&barreira,NULL,NTHREADS);

  for(calls=0; calls < TOTAL_CALLS; calls++)
  {
    generations = 1;

    for(t=0; t<NTHREADS; t++) {
      #ifdef DEBUG
        printf("--Cria a thread %d\n", t);
      #endif
      if (pthread_create(&tid[t], NULL, ThreadRun, (void *)t)) {
        printf("--ERRO: pthread_create()\n"); exit(-1);
      }
    }

    for (t=0; t<NTHREADS; t++) {
      if (pthread_join(tid[t], NULL)) {
           printf("--ERRO: pthread_join() \n"); exit(-1); 
      } 
    }
    
    //reduce para o processo #1 do melhor individuo
    MPI_Gather( currentGeneration, 10, MPI_DOUBLE, bestGeneration, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    //faz a migração
    MPI_Scatter( bestGeneration, 10, MPI_DOUBLE, &currentGeneration[1], 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
}

int totalDeRestricoesQuebradas(Individuo individuo)
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
  
  constraint = min(0, individuo.x[0] - 100);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0, 10000 - individuo.x[0]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0, individuo.x[1] - 1000);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0, 10000 - individuo.x[1]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0, individuo.x[2] - 1000);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  constraint = min(0, 10000 - individuo.x[2]);
  if (constraint < 0)
  {
    restricoesQuebradas++;
  }
  
  for(gene=3; gene<8; gene++)
  {
    constraint = min(0, individuo.x[gene] - 10);
    if (constraint < 0)
    {
      restricoesQuebradas++;
    }
  
    constraint = min(0, 1000 - individuo.x[gene]);
    if (constraint < 0)
    {
      restricoesQuebradas++;
    }
  }
  
  return restricoesQuebradas;
}

void exibirRestricoesQuebradas(Individuo individuo)
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
  
  constraint = min(0, individuo.x[0] - 100);
  if (constraint < 0)
  {
    printf("Restricao 100 <= x1 não respeitada.\n");
  }
  
  constraint = min(0, 10000 - individuo.x[0]);
  if (constraint < 0)
  {
    printf("Restricao x1 <= 10000 não respeitada.\n");
  }
  
  constraint = min(0, individuo.x[1] - 1000);
  if (constraint < 0)
  {
    printf("Restricao 1000 <= x2 não respeitada.\n");
  }
  
  constraint = min(0, 10000 - individuo.x[1]);
  if (constraint < 0)
  {
    printf("Restricao x2 <= 10000 não respeitada.\n");
  }
  
  constraint = min(0, individuo.x[2] - 1000);
  if (constraint < 0)
  {
    printf("Restricao 1000 <= x3 não respeitada.\n");
  }
  
  constraint = min(0, 10000 - individuo.x[2]);
  if (constraint < 0)
  {
    printf("Restricao x3 <= 10000 não respeitada.\n");
  }
  
  for(gene=3; gene<8; gene++)
  {
    constraint = min(0, individuo.x[gene] - 10);
    if (constraint < 0)
    {
      printf("Restricao 10 <= x%d não respeitada.\n", gene+1);
    }
  
    constraint = min(0, 1000 - individuo.x[gene]);
    if (constraint < 0)
    {
      printf("Restricao x%d <= 1000 não respeitada.\n", gene+1);
    }
  }
}

void exibirErroOtimo(Individuo individuo)
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

int main(int argc, char **argv)
{
  int organism, gene;
  int restricoesQuebradas, menorNumeroRestricoesQuebradas;
  int melhorIndividuo;
  struct timeval inicio, fim;
  double tempo;
  
  /* initialize random seed: */
  srand(time(NULL));

  /* Initialize MPI */
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // Alocar um vetor para a melhor populacao no processo #1
  bestGeneration = (Individuo*)malloc(sizeof(Individuo) * nprocs);

  AllocateMemory();
  
  InitializeOrganisms();
 
  gettimeofday(&inicio, NULL); 
  Run();
  gettimeofday(&fim, NULL);
  
  if (rank == 0)
  {
    tempo = (fim.tv_sec - inicio.tv_sec)*1000 + (fim.tv_usec - inicio.tv_usec)/1000; //calcula tempo em milisegundos
    printf("Tempo por geração: %.1lf(ms)\n", tempo/TOTAL_GENERATIONS);
    printf("Tempo total: %.3lf(ms)\n", tempo);

    melhorIndividuo = 0;
    menorNumeroRestricoesQuebradas = totalDeRestricoesQuebradas(bestGeneration[0]);
    
    for(organism=0; organism<nprocs; organism++)
    {
      restricoesQuebradas = totalDeRestricoesQuebradas(bestGeneration[organism]);
      if (restricoesQuebradas < menorNumeroRestricoesQuebradas)
      {
        menorNumeroRestricoesQuebradas = restricoesQuebradas;
        melhorIndividuo = organism;
      }
    }

    printf("Melhor fitness: %.2f\n", bestGeneration[melhorIndividuo].fitness);

    for(gene=0; gene<8; gene++)
    {
      printf("X%d: %.2f\n", gene+1, bestGeneration[melhorIndividuo].x[gene]);
    }
    
    printf("Restricoes quebradas:\n");
    exibirRestricoesQuebradas(bestGeneration[melhorIndividuo]);
    printf("Total: %d\n", menorNumeroRestricoesQuebradas);
    
    exibirErroOtimo(bestGeneration[melhorIndividuo]);
  }

  MPI_Finalize();

  return 0;
}
