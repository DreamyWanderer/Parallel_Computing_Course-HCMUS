#include <stdio.h>

#define ARRAYSIZE(x) ( sizeof(x)/sizeof(*x) )

void printDeviceInfo()
{
    cudaDeviceProp prop;

    cudaGetDeviceProperties(&prop, 0);

    printf("GPU card's name: %s\n", prop.name);
    printf("GPU computation capabilities: %d.%d\n", prop.major, prop.minor);
    printf("Maximum number of block dimensions: %lu dimensions\n", ARRAYSIZE(prop.maxThreadsDim) );
    printf("Maximum size of each block dimensions: %d.x %d.y %d.z\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
    printf("Maximum number of grid dimensions: %lu dimesions\n", ARRAYSIZE(prop.maxGridSize) );
    printf("Maximum size of each grid dimensions: %d.x %d.y %d.y\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
    printf("Maximum size of GPU memory: %zu bytes\n", prop.totalGlobalMem);
    printf("Size of constant memory: %zu bytes\n", prop.totalConstMem);
    printf("Size of shared memory per multiprocessor and in GPU device: %zu bytes and %zu bytes\n", prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor * prop.multiProcessorCount);
}

int main(int argc, char** argv)
{
    printf("Hello World, I made by a person named The Hoang!\n\n");

    printDeviceInfo();

    return 0;
}