#include <stdio.h>

constexpr int BLOCK_SIZE = 256;
float time_host = 0;

#define ARRAYSIZE(x) ( sizeof(x)/sizeof(*x) )
#define CUDA_CHECK(call)\
{\
    cudaError_t err = call;\
    \
    if (err != cudaSuccess)\
    {\
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);\
        exit(EXIT_FAILURE);\
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

__global__ void addVecKernelV1(int N, float *d_A, float *d_B, float *d_C)
{
    int i_1 = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int i_2 = i_1 + blockDim.x;

    if (i_1 < N)
    {
        d_C[i_1] = d_A[i_1] + d_B[i_1];
        //printf("Calculating %d and %d\n", i_1, i_2);
    }
    if (i_2 < N)
    {
        d_C[i_2] = d_A[i_2] + d_B[i_2];
    }
    /*
    if (i_1 >= N || i_1 >= N)
    {
        printf("Calculating %d and %d\n", i_1, i_2);
    }
    */
}

__global__ void addVecKernelV2(int N, float *d_A, float *d_B, float *d_C)
{
    int i_1 = (blockIdx.x * blockDim.x  + threadIdx.x) * 2;
    int i_2 = i_1 + 1;

    if (i_1 < N)
    {
        d_C[i_1] = d_A[i_1] + d_B[i_1];
    }
    if (i_2 < N)
    {
        d_C[i_2] = d_A[i_2] + d_B[i_2];
    }
    /*
    if (i_1 >= N || i_1 >= N)
    {
        printf("Calculating %d and %d\n", i_1, i_2);
    }
    */
}

void addVecCPU(int N, float *A, float *B, float *C)
{
    GpuTimer timer;
    time_host = 0;

    timer.Start();
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
    timer.Stop();
    time_host = timer.Elapsed();
}

void addVecCPUGPU(int N, float *A, float *B, float *C)
{
    GpuTimer timer;
    float *d_A, *d_B, *d_C;
    const int sizeVecByte = N * sizeof(float);

    CUDA_CHECK( cudaMalloc(&d_A, sizeVecByte) );
    CUDA_CHECK( cudaMalloc(&d_B, sizeVecByte) );
    CUDA_CHECK( cudaMalloc(&d_C, sizeVecByte) );

    CUDA_CHECK( cudaMemcpy(d_A, A, sizeVecByte, cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(d_B, B, sizeVecByte, cudaMemcpyHostToDevice) );

    // First version experiment from this point
    timer.Start();
    addVecKernelV1<<<ceil(N/2/float(BLOCK_SIZE)), BLOCK_SIZE>>>(N, d_A, d_B, d_C); 
    cudaDeviceSynchronize();
    timer.Stop();
    CUDA_CHECK(cudaGetLastError());
    float time_1 = timer.Elapsed();
    // Second version experiment starts from this point
    timer.Start();
    addVecKernelV2<<<ceil(N/2/float(BLOCK_SIZE)), BLOCK_SIZE>>>(N, d_A, d_B, d_C); 
    cudaDeviceSynchronize();
    timer.Stop();
    CUDA_CHECK(cudaGetLastError());
    float time_2 = timer.Elapsed();
    // End all experiments here

    CUDA_CHECK( cudaMemcpy(C, d_C, sizeVecByte, cudaMemcpyDeviceToHost) );

    CUDA_CHECK( cudaFree(d_A) );
    CUDA_CHECK( cudaFree(d_B) );
    CUDA_CHECK( cudaFree(d_C) );

    printf("|\t%-10d\t|\t%-10f\t|\t%-20f\t|\t%-20f|\n", N, time_host, time_1, time_2);
}

void DoExperimentWithSizeN(int N)
{
    // From here, this part is taken from Demo file which is given by teacher

    float *in1, *in2; // Input vectors
    float *out;  // Output vector

    // Allocate memories for in1, in2, out
    size_t nBytes = N * sizeof(float);
    in1 = (float *)malloc(nBytes);
    in2 = (float *)malloc(nBytes);
    out = (float *)malloc(nBytes);

    // Input data into in1, in2
    for (int i = 0; i < N; i++)
    {
    	in1[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    	in2[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    addVecCPU(N, in1, in2, out);
    addVecCPUGPU(N, in1, in2, out);

    free(in1);
    free(in2);
    free(out);
}

int main(int argc, char **argv)
{
    int listN[10] = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216};

    printf("|     Vector Size     |       Host time       |       Device time (version 1)    |    Device time (version 2)    |\n");

    for (int i = 0; i < ARRAYSIZE(listN); i++)
    {
        DoExperimentWithSizeN(listN[i]);
    }
}