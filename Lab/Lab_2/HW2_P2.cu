#include <stdio.h>

//#define idx1D(r, c, colSz) r * colSz + c // DAMN IT MACRO

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

__host__ __device__ int idx1D(int r, int c, int colSz) // Create two verision: __host__ to be callable from CPU and run on CPU, __device__ to be callable from GPU and run on GPU
{
    return r * colSz + c;
}

// Run first to remove overhead when run this program in the first time
__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

__global__ void matrix_multiplication_kernel1(float* A, float* B, float* C, int m, int n, int k)
{
	//TODO
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    int globalCIdx = idx1D(tidY, tidX, k);

    for (int i = 0; i < n; i++)
    {
        C[globalCIdx] += A[idx1D(tidY, i, n)] * B[idx1D(i, tidX, k)];
        //if (tidY == 0 && tidX == 0) printf("%f * %f\n", A[idx1D(tidY, i, n)], B[idx1D(i, tidX, k)]); // Debug: print A and B when calculating normally by device
    }
}

__global__ void matrix_multiplication_kernel2(float* A, float* B, float* C, int m, int n, int k)
{
	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	//TODO

    int numStride = (n - 1) / TILE_WIDTH + 1;
    int tidY = blockIdx.y * blockDim.y + threadIdx.y;
    int tidX = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = 0; stride < numStride; stride++)
    {   
        int globalAIdx = idx1D(tidY, stride * TILE_WIDTH + threadIdx.x, n);
        int globalBIdx = idx1D(stride * TILE_WIDTH + threadIdx.y, tidX, k);
        
        // Debug: print position we will get from A
        //if (tidY == 0 & tidX == 0) printf("%d\n", (stride * TILE_WIDTH + threadIdx.y) * k + tidX);

        //if (tidY == 0 & tidX == 0) printf("A %d %f\n", globalAIdx, A[globalAIdx]);

        if (globalAIdx < m * n)
            s_A[threadIdx.y][threadIdx.x] = A[globalAIdx];
        else
            s_A[threadIdx.y][threadIdx.x] = 0;

        if (tidY == 0 & tidX == 0) printf("B %d %f\n", globalBIdx, B[globalBIdx]);

        if (globalBIdx < n * k)
            s_B[threadIdx.y][threadIdx.x] = B[globalBIdx];
        else
            s_B[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        /* CHECKED: sub matrix and original matrix are the same
        // Print whole s_B
        if (tidY == 0 && tidX == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        {
            printf("s_B %d:\n", stride);
            for (int i = 0; i < TILE_WIDTH; i++)
            {
                for (int j = 0; j < TILE_WIDTH; j++)
                    printf("%f ", s_B[i][j]);
                printf("\n");
            }
            printf("\n");
        }
        __syncthreads();
        if (tidY == 0 && tidX == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        {
            printf("B %d:\n", stride);
            for (int i = 0; i < TILE_WIDTH; i++)
            {
                for (int j = 0; j < TILE_WIDTH; j++)
                    printf("(%d %d) %f ",stride * TILE_WIDTH + i, tidX + j, B[idx1D(stride * TILE_WIDTH + i, tidX + j, k)]);
                printf("\n");
            }
            printf("\n");
        }
        __syncthreads();
        */
        
    
        for (int i = 0; i < TILE_WIDTH; i++)
        {
            C[idx1D(tidY, tidX, k)] += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
            //if (tidY == 0 & tidX == 0) printf("%f * %f\n", s_A[threadIdx.y][i], s_B[i][threadIdx.x]); // Debug: check what we are using to calculate an element of C
        }

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
                for (int i = 0; i < n; i++) 
                {
                    C[idx1D(r, c, k)] += A[idx1D(r, i, n)] * B[idx1D(i, c, k)];
                    if (r == 0 & c == 0) printf("%f * %f\n", A[idx1D(r, i, n)], B[idx1D(i, c, k)]);
                }
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

        dim3 gridSize( ( m - 1)/(blockSize.y) + 1, (k - 1)/(blockSize.x) + 1); // TODO: Compute gridSize

        // Run the warmup process
        timer.Start();
        warm_up_gpu<<<gridSize, blockSize>>>();
        cudaDeviceSynchronize();
        timer.Stop();
        float warmUpTime = timer.Elapsed();
        printf("Warm up time: %f ms\n", warmUpTime);
        
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

    for (int r = 0; r < 64; r++) printf("%d %f ", idx1D(r, 0, k), h_B[idx1D(r, 0, k)]); printf("\n");

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
    printf("A[0][0] = %f\n", h_C[0]);
	printf("Error between device result and host result: %f\n\n", err);

    // THANKS COPILOT SINCE YOU DO NOT SHOW ME THIS DAMN ERROR
    memset(h_C, 0.0, m * k * sizeof(float));

	printf("Shared memory Matrix Multiplication:\n");
    matrix_multiplication(h_A, h_B, h_C, m, n, k, true,blockSize,2);
	err = checkCorrectness(h_C, correct_C,m*k);
    printf("A[0][0] = %f\n", h_C[0]);
	printf("Error between device result and host result: %f", err);	

    // Print whole C
    /*
    printf("C:\n");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0;j < 10;j++)
            printf("%f ", h_C[i*k+j]);
        printf("\n");
    }
    printf("\n");

    // Print whole correct_C
    printf("correct_C:\n");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0;j < 10;j++)
            printf("%f ", correct_C[i*k+j]);
        printf("\n");
    }
    */
	
    free(h_A);
    free(h_B);
    free(h_C);
    free(correct_C);

    return 0;
}
