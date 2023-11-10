#include <stdio.h>

#define idx1D(r, c, colSz) r * colSz + c

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}
#define TILE_WIDTH 32
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

__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k)
{
	//TODO
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCIdx = idx1D(tidX, tidY, k);

    for (int i = 0; i < n; i++)
    {
        C[globalCIdx] += A[idx1D(tidX, i, n)] + B[idx1D(i, tidY, k)];
    }
}

__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	//TODO

    int numStride = (n - 1) / TILE_WIDTH + 1;
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;

    for (int stride = 0; stride <= numStride; stride++)
    {
        int globalAIdx = idx1D(tidX, stride * TILE_WIDTH + threadIdx.y, n);
        int globalBIdx = idx1D(stride * TILE_WIDTH + threadIdx.x, tidY, k);

        if (globalAIdx < m * n)
            s_A[threadIdx.x][threadIdx.y] = A[globalAIdx];
        else
            s_A[threadIdx.x][threadIdx.y] = 0;

        if (globalBIdx < n * k)
            s_B[threadIdx.x][threadIdx.y] = B[globalBIdx];
        else
            s_B[threadIdx.x][threadIdx.y] = 0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; i++)
            C[idx1D(tidX, tidY, k)] += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];

        __syncthreads();
    }
}

void matrix_multiplication(float* A, float* B, float* C, int m, int n, int k,
    bool useDevice = false, dim3 blockSize = dim3(1),int kernelType=1)
{
    GpuTimer timer;
    timer.Start();
    if (useDevice == false)
    {
        for (int r = 0; r < m; r++)
        {
            for (int c = 0; c < k; c++)
            {
                for (int i = 0; i < n; i++) C[idx1D(r, c, k)] += A[idx1D(r, i, n)] + B[idx1D(i, c, k)];
            }
        }
    }
    else // Use device
    {
        // TODO: Allocate device memories
        float* d_A, * d_B, * d_C;
        const int sizeVecA = sizeof(float) * m * n;
        const int sizeVecB = sizeof(float) * n * k;
        const int sizeVecC = sizeof(float) * m *k;
        CHECK( cudaMalloc(&d_A, sizeVecA));
        CHECK( cudaMalloc(&d_B, sizeVecB));
        CHECK( cudaMalloc(&d_C, sizeVecC));

        // TODO: Copy data to device memories
        CHECK( cudaMemcpy(d_A, A, sizeVecA, cudaMemcpyHostToDevice));
        CHECK( cudaMemcpy(d_B, B, sizeVecB, cudaMemcpyHostToDevice));
        CHECK( cudaMemcpy(d_C, C, sizeVecC, cudaMemcpyHostToDevice));

        dim3 gridSize( ( m - 1)/(blockSize.x) + 1, (k - 1)/(blockSize.y) + 1); // TODO: Compute gridSize
        
		if (kernelType == 1)
			matrix_multiplication_kernel1<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);
		else if (kernelType == 2)
			matrix_multiplication_kernel2<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, n, k);

        // TODO: Copy result from device memory
        CHECK( cudaMemcpy(C, d_C, sizeVecC, cudaMemcpyDeviceToHost));

        // TODO: Free device memories
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
		
		printf("Grid size: %d * %d, block size: %d * %d\n", 
			gridSize.x,gridSize.y, blockSize.x,blockSize.y);

    }
    timer.Stop();
    float time = timer.Elapsed();
    printf("Processing time (%s): %f ms\n",
        useDevice == true ? "use device" : "use host", time);
}

float checkCorrectness(float * a1, float* a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)	
		err += abs(a1[i] - a2[i]);
	err /= n;
	return err;
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
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");

}
int main(int argc, char** argv)
{
	printDeviceInfo();
	
	//Declare variables
    float* h_A; // The A matrix
    float* h_B; // The B matrix
    float* h_C; // The output C matrix
    float* correct_C; // The output C matrix

    int m;    // number of rows in the matrix A
    int n; // number of columns in the matrix A, number of rows in the matrix B
    int k; // number of columns in the matrix B

    m = (1 << 10);
    n = (1 << 9);
    k = (1 << 10);

    // Set up input data
    h_A = (float*)malloc(m * n * sizeof(float));
    h_B = (float*)malloc(n * k * sizeof(float));
    h_C = (float*)malloc(m * k * sizeof(float));
    correct_C = (float*)malloc(m * k * sizeof(float));

    for (int i = 0; i < m; i++)
        for (int j = 0;j < n;j++)
            h_A[i*n+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
 
    for (int i = 0; i < n; i++)
        for (int j = 0;j < k;j++)
            h_B[i*k+j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);


    // Add vectors (on host)
    matrix_multiplication(h_A,h_B,correct_C,m,n,k);
	printf("\n");

	dim3 blockSize(32, 32); // Default
	if (argc == 3)
	{
		blockSize.x = atoi(argv[1]);
		blockSize.y = atoi(argv[2]);
	} 
    // Add in1 & in2 on device
	printf("Basic Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,1);
	float err = checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f\n\n", err);

	printf("Shared memory Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,2);
	err = checkCorrectness(h_C, correct_C,m*k);
	printf("Error between device result and host result: %f", err);	
	
    free(h_A);
    free(h_B);
    free(h_C);
    free(correct_C);

    return 0;
}
