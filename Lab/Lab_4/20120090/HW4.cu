#include <stdio.h>
#include <stdint.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(1);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential Radix Sort
void sortByHost(const uint32_t * in, int n,
                uint32_t * out)
{
    int * bits = (int *)malloc(n * sizeof(int));
    int * nOnesBefore = (int *)malloc(n * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // To free memory later
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * dst = out;

    // Loop from LSB (Least Significant Bit) to MSB (Most Significant Bit)
	// In each loop, sort elements according to the current bit from src to dst 
	// (using STABLE counting sort)
    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx++)
    {
        // Extract bits
        for (int i = 0; i < n; i++)
            bits[i] = (src[i] >> bitIdx) & 1;

        // Compute nOnesBefore
        nOnesBefore[0] = 0;
        for (int i = 1; i < n; i++)
            nOnesBefore[i] = nOnesBefore[i-1] + bits[i-1];

        // Compute rank and write to dst
        int nZeros = n - nOnesBefore[n-1] - bits[n-1];
        for (int i = 0; i < n; i++)
        {
            int rank;
            if (bits[i] == 0)
                rank = i - nOnesBefore[i];
            else
                rank = nZeros + nOnesBefore[i];
            dst[rank] = src[i];
        }

        // Swap src and dst
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    // Does out array contain results?
    memcpy(out, src, n * sizeof(uint32_t));

    // Free memory
    free(originalSrc);
    free(bits);
    free(nOnesBefore);
}

__device__ int blockCount = 0;
volatile __device__ int volatileBlockCount = 0;

__global__ void prefixSumKernel(uint32_t* input, int size, uint32_t* output, volatile uint32_t* blockSums){
    // Shared memory for the block
    extern __shared__ uint32_t sharedData[];
    __shared__ int blockIndex;

    // Only one thread updates the block index
    if(threadIdx.x == 0)
        blockIndex = atomicAdd(&blockCount, 1);
    __syncthreads();

    // Calculate the indices for the input array
    int index1 = blockIndex * 2 * blockDim.x + threadIdx.x;
    int index2 = index1 + blockDim.x;

    // Load data into shared memory
    sharedData[threadIdx.x] = (0 < index1 && index1 < size) ? input[index1 - 1] : 0;
    sharedData[threadIdx.x + blockDim.x] = (index2 < size) ? input[index2 - 1] :0;
    __syncthreads();

    // Perform the first phase of the scan in shared memory
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2){
        int sharedDataIndex = (threadIdx.x + 1) * 2 * stride - 1;
        if (sharedDataIndex < 2 * blockDim.x)
            sharedData[sharedDataIndex] += sharedData[sharedDataIndex - stride];
        __syncthreads();
    }

    // Perform the second phase of the scan in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2){
        int sharedDataIndex = (threadIdx.x + 1) * 2 * stride - 1 + stride;
        if (sharedDataIndex < 2 * blockDim.x)
            sharedData[sharedDataIndex] += sharedData[sharedDataIndex - stride];
        __syncthreads();
    }

    // Write the block sum to the blockSums array
    if (blockSums != NULL && threadIdx.x == 0)
        blockSums[blockIndex] = sharedData[2 * blockDim.x - 1];
    __syncthreads();

    // Wait for all blocks to finish
    if(threadIdx.x == 0){
        if(blockIndex > 0){
            while(volatileBlockCount < blockIndex){}
            blockSums[blockIndex] += blockSums[blockIndex - 1];
            __threadfence();
        }
        volatileBlockCount += 1;
    }
    __syncthreads();

    // Calculate block sum
    if (index1 < size)
        output[index1] = sharedData[threadIdx.x] + ((blockIndex > 0) ? blockSums[blockIndex - 1] : 0);
    if (index2 < size)
        output[index2] = sharedData[threadIdx.x + blockDim.x] + ((blockIndex > 0) ? blockSums[blockIndex - 1] : 0);
}

__global__ void calculateRankAndReorder(uint32_t *source, uint32_t *destination, uint32_t *bitValues, uint32_t *onesCountBefore, int zeroCount, int total){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < total){
        int rank = (bitValues[index] == 0) ? index - onesCountBefore[index] : zeroCount + onesCountBefore[index];
        destination[rank] = source[index];
    }
}

__global__ void isolateBits(uint32_t *inputData, uint32_t *outputData, int bitPosition, int total){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < total)
        outputData[index] = (inputData[index] >> bitPosition) & 1;
}

// Parallel Radix Sort
void sortByDevice(const uint32_t * in, int n, uint32_t * out, int blockSize)
{
    // TODO
    uint32_t *d_src, *d_dst; // For ranking and swapping 
    uint32_t *d_bits, *d_nOnesBefore; // d_bits for bits, d_nOnesBefore for exclusive scanned bits
    uint32_t *d_blkSums;
    size_t nBytes = n * sizeof(uint32_t);
    const int z = 0;
    uint32_t totalOnes;
    uint32_t lastBit;
    CHECK(cudaMalloc(&d_bits, nBytes));
    CHECK(cudaMalloc(&d_nOnesBefore, nBytes));

    CHECK(cudaMalloc(&d_src, nBytes));
    CHECK(cudaMalloc(&d_dst, nBytes));
    CHECK(cudaMemcpy(d_src, in, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_dst, out, nBytes, cudaMemcpyHostToDevice));
    
    dim3 gridSize((n - 1)/blockSize + 1);
    dim3 gridSize1((n - 1)/blockSize/2 + 1);
    if (gridSize.x > 1){
        CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(uint32_t)));
    }
    else
        d_blkSums = NULL;

    for (int bitIdx = 0; bitIdx < sizeof(uint32_t) * 8; bitIdx ++){
        CHECK(cudaMemcpyToSymbol(blockCount , &z, sizeof(int)));
        CHECK(cudaMemcpyToSymbol(volatileBlockCount, &z, sizeof(int)));

        isolateBits<<<gridSize,blockSize>>>(d_src, d_bits, bitIdx, n);

        size_t smem = 2 * blockSize * sizeof(uint32_t);
        
        prefixSumKernel<<<gridSize1,blockSize,smem>>>(d_bits, n, d_nOnesBefore, d_blkSums);
        
        CHECK(cudaMemcpy(&lastBit, d_bits + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&totalOnes, d_nOnesBefore + n - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        int nZeros = n - totalOnes - lastBit;
        calculateRankAndReorder<<<gridSize,blockSize>>>(d_src, d_dst, d_bits, d_nOnesBefore, nZeros, n);
        CHECK(cudaGetLastError());
        uint32_t * temp = d_src;
        d_src = d_dst;
        d_dst = temp;
    }
    CHECK(cudaMemcpy(out, d_src, nBytes, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_nOnesBefore));
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_blkSums));
}

// Radix Sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        bool useDevice=false, int blockSize=1)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix Sort by host\n");
        sortByHost(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix Sort by device\n");
        sortByDevice(in, n, out, blockSize);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    //int n = 50; // For test by eye
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
    {
        //in[i] = rand() % 255; // For test by eye
        in[i] = rand();
    }
    //printArray(in, n); // For test by eye

    // DETERMINE BLOCK SIZE
    int blockSize = 512; // Default 
    if (argc == 2)
        blockSize = atoi(argv[1]);

    // SORT BY HOST
    sort(in, n, correctOut);
    //printArray(correctOut, n); // For test by eye
    
    // SORT BY DEVICE
    sort(in, n, out, true, blockSize);
    //printArray(out, n); // For test by eye
    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
