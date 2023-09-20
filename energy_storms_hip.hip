/*
 * Simplified simulation of high-energy particle storms
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2017/2018
 *
 * Version: 2.0
 *
 * Code prepared to be used with the Tablon on-line judge.
 * The current Parallel Computing course includes contests using:
 * OpenMP, MPI, and CUDA.
 *
 * (c) 2018 Arturo Gonzalez-Escribano, Eduardo Rodriguez-Gutiez
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the HIP assignment versions */
#include<hip/hip_runtime.h>

/*
 * Macros to show errors when calling a HIP library function,
 * or after launching a kernel
 */
#define CHECK_HIP_CALL( a )	{ \
	hipError_t ok = a; \
	if ( ok != hipSuccess ) \
		fprintf(stderr, "-- Error HIP call in line %d: %s\n", __LINE__, hipGetErrorString( ok ) ); \
	}
#define CHECK_HIP_LAST()	{ \
	hipError_t ok = hipGetLastError(); \
	if ( ok != hipSuccess ) \
		fprintf(stderr, "-- Error HIP last in line %d: %s\n", __LINE__, hipGetErrorString( ok ) ); \
	}


/* Use fopen function in local tests. The Tablon online judge software 
   substitutes it by a different function to run in its sandbox */
#ifdef CP_TABLON
#include "cputilstablon.h"
#else
#define    cp_open_file(name) fopen(name,"r")
#endif

/* Function to get wall time */
double cp_Wtime(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}


#define THRESHOLD    0.001f
#define NTHREADS 256

/* Structure used to store data for one storm of particles */
typedef struct {
    int size;    // Number of particles
    int *posval; // Positions and values
} Storm;

typedef struct {
    float value;
    int index;
} ValIndexPair;

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
__device__ void update(float *layer, int layer_size, int k, int pos, float energy) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    if ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size )
        layer[threadIdx.x] = layer[threadIdx.x] + energy_k;
}


/* ANCILLARY FUNCTIONS: These are not called from the code section which is measured, leave untouched */
/* DEBUG function: Prints the layer status */
void debug_print(int layer_size, float *layer, int *positions, float *maximum, int num_storms ) {
    int i,k;
    /* Only print for array size up to 35 (change it for bigger sizes if needed) */
    if ( layer_size <= 35 ) {
        /* Traverse layer */
        for( k=0; k<layer_size; k++ ) {
            /* Print the energy value of the current cell */
            printf("%10.4f |", layer[k] );

            /* Compute the number of characters. 
               This number is normalized, the maximum level is depicted with 60 characters */
            int ticks = (int)( 60 * layer[k] / maximum[num_storms-1] );

            /* Print all characters except the last one */
            for (i=0; i<ticks-1; i++ ) printf("o");

            /* If the cell is a local maximum print a special trailing character */
            if ( k>0 && k<layer_size-1 && layer[k] > layer[k-1] && layer[k] > layer[k+1] )
                printf("x");
            else
                printf("o");

            /* If the cell is the maximum of any storm, print the storm mark */
            for (i=0; i<num_storms; i++) 
                if ( positions[i] == k ) printf(" M%d", i );

            /* Line feed */
            printf("\n");
        }
    }
}

/*
 * Function: Read data of particle storms from a file
 */
Storm read_storm_file( char *fname ) {
    FILE *fstorm = cp_open_file( fname );
    if ( fstorm == NULL ) {
        fprintf(stderr,"Error: Opening storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    Storm storm;    
    int ok = fscanf(fstorm, "%d", &(storm.size) );
    if ( ok != 1 ) {
        fprintf(stderr,"Error: Reading size of storm file %s\n", fname );
        exit( EXIT_FAILURE );
    }

    storm.posval = (int *)malloc( sizeof(int) * storm.size * 2 );
    if ( storm.posval == NULL ) {
        fprintf(stderr,"Error: Allocating memory for storm file %s, with size %d\n", fname, storm.size );
        exit( EXIT_FAILURE );
    }
    
    int elem;
    for ( elem=0; elem<storm.size; elem++ ) {
        ok = fscanf(fstorm, "%d %d\n", 
                    &(storm.posval[elem*2]),
                    &(storm.posval[elem*2+1]) );
        if ( ok != 2 ) {
            fprintf(stderr,"Error: Reading element %d in storm file %s\n", elem, fname );
            exit( EXIT_FAILURE );
        }
    }
    fclose( fstorm );

    return storm;
}

typedef union  {
    ValIndexPair pair;
    unsigned long long int ulong;
} ValIndexPairAtomic;

__device__ void atomicMax(ValIndexPair* address, float val, int index)
{
    ValIndexPairAtomic test;
    test.pair.value = val;
    test.pair.index = index;

    ValIndexPairAtomic current;
    current.pair = *address;
    while (val > current.pair.value) {
        current.ulong = atomicCAS((unsigned long long int*)address, current.ulong, test.ulong);
    }
}

__global__ void kernel(Storm *storms, int num_storms, int layer_size, ValIndexPair *maximum) {
    __shared__ float layer1_alloc[NTHREADS];
    __shared__ float layer2_alloc[NTHREADS];
    float* layer1 = layer1_alloc;
    float* layer2 = layer2_alloc;
    
    assert(blockDim.x == NTHREADS);
    int k = blockIdx.x * (blockDim.x - 2) + threadIdx.x;
    int tid = threadIdx.x;
    if (k >= layer_size)
        return;

    layer1[tid] = 0.0f;

    /* 4. Storms simulation */
    for(int i = 0; i < num_storms; i++) {
        /* 4.1. Add impacts energies to layer cells */
        for (int j = 0; j < storms[i].size; j++) {
            int position = storms[i].posval[j*2];
            float energy = (float)storms[i].posval[j*2+1] * 1000;
            update(layer1, layer_size, k, position, energy);
        }

        __syncthreads();
        if (tid == 0 || tid == NTHREADS - 1 || k == layer_size - 1) {
            layer2[tid] = layer1[tid];
        } else {
            /* 4.2. Energy relaxation between storms */
            /* 4.2.2. Update layer using the ancillary values.
                    Skip updating the first and last positions */
            layer2[tid] = (layer1[tid-1] + layer1[tid] + layer1[tid+1]) / 3;
        }

        __syncthreads();
        if (!(tid == 0 || tid == NTHREADS - 1 || k == layer_size - 1)) {
            /* 4.3. Locate the maximum value in the layer, and its position */
            /* Check it only if it is a local maximum */
            if (layer2[tid] > layer2[tid-1] && layer2[tid] > layer2[tid+1] ) {
                atomicMax(&maximum[i], layer2[tid], k);
            }
        }

        // swap layers 
        float* tmp = layer1;
        layer1 = layer2;
        layer2 = tmp;
    }
}


/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    Storm storms[ num_storms ];

    /* 1.2. Read storms information */
    for(int i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );

    /* 1.3. Intialize maximum levels to zero */
    ValIndexPair maximum[num_storms];

    /* 2. Begin time measurement */
    CHECK_HIP_CALL( hipSetDevice(0) );
    CHECK_HIP_CALL( hipDeviceSynchronize() );
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */

    /* 3.1 Allocate memory for the result */
    ValIndexPair *d_maximum = NULL;
    CHECK_HIP_CALL(hipMalloc(&d_maximum, sizeof(ValIndexPair) * num_storms));
    CHECK_HIP_CALL(hipMemset(d_maximum, 0, sizeof(ValIndexPair) * num_storms));

    /* 3.2 Allocate memory for the input data `storms` and copy the data to the device */
    Storm *d_storms = NULL;
    CHECK_HIP_CALL(hipMalloc(&d_storms, sizeof(Storm) * num_storms));
    CHECK_HIP_CALL(hipMemcpy(d_storms, storms, sizeof(Storm) * num_storms, hipMemcpyHostToDevice));
    for (int i = 0; i < num_storms; i++) {
        CHECK_HIP_CALL(hipMalloc(&(d_storms[i].posval), sizeof(int) * storms[i].size * 2));
        CHECK_HIP_CALL(hipMemcpy(
            d_storms[i].posval, 
            storms[i].posval, 
            sizeof(int) * storms[i].size * 2, 
            hipMemcpyHostToDevice));
    }

    dim3 threadsPerBlock(NTHREADS);
    dim3 blocks((layer_size - 3) / (NTHREADS - 2) + 1);
    hipLaunchKernelGGL(
        kernel,
        blocks, threadsPerBlock, 0, 0,
        d_storms, num_storms, layer_size, d_maximum);
    
    /* 4.4. Copy the results from device*/
    CHECK_HIP_CALL(hipMemcpy(
            maximum, 
            d_maximum, 
            sizeof(ValIndexPair) * num_storms, 
            hipMemcpyDeviceToHost));

    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    CHECK_HIP_CALL( hipDeviceSynchronize() );
    ttotal = cp_Wtime() - ttotal;

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG
    //debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
    /* 7.2. Print the maximum levels */
    printf("Result:");
    for (int i = 0; i < num_storms; i++)
        printf(" %d %f", maximum[i].index, maximum[i].value);
    printf("\n");

    /* 8. Free resources */    
    for(int i = 0; i < num_storms; i++) {
        free(storms[i].posval);
        hipFree(d_storms[i].posval);
    }

    hipFree(d_storms);
    hipFree(d_maximum);

    /* 9. Program ended successfully */
    return 0;
}

