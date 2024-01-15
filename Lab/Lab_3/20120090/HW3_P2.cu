#include <stdio.h>

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
		cudaEventRecord(start,0);
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

__global__ void addVecKernel(int *in1, int *in2, int n, 
        int *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 

    if (i < n)
    {
        out[i] = in1[i] + in2[i];
    }
}

void addVec(int *in1, int *in2, int n, 
        int *out, 
        bool useDevice=false, dim3 blockSize=dim3(1), int nStreams=1)
{
	if (useDevice == false)
	{
        for (int i = 0; i < n; i++)
        {
            out[i] = in1[i] + in2[i];
        }
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);
        
        size_t nBytes = n * sizeof(int);

		// TODO: Allocate nStreams device memory regions
		float **d_vectors;
		cudaStream_t *streams;
		d_vectors = (float**) malloc(nStreams * sizeof(float*));
		streams = (cudaStream_t*) malloc(nStreams * sizeof(cudaStream_t));

        // TODO: Create "nStreams" device streams
		for (int i = 0; i < nStreams; i++)
		{
			cudaMalloc((void**)&d_vectors[i], n * sizeof(float));
			cudaStreamCreate(&streams[i]);
		}

        GpuTimer timer;
        timer.Start();

        // TODO: Send jobs (H2D, kernel, D2H) to device streams run the jobs in parallel
		int streamSize = n / nStreams;
		int streamBytes = streamSize * sizeof(float);
		for (int i = 0; i < nStreams; i++)
		{
			int offset = i * streamSize;
			cudaMemcpyAsync(&d_vectors[i][offset], &in1[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
			cudaMemcpyAsync(&d_vectors[i][offset], &in2[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]);
			addVecKernel<<<(streamSize + blockSize.x - 1) / blockSize.x, blockSize.x, 0, streams[i]>>>((int*)&d_vectors[i][offset], (int*)&d_vectors[i][offset], streamSize, (int*)&d_vectors[i][offset]);
			cudaMemcpyAsync(&out[offset], &d_vectors[i][offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]);
		}

        timer.Stop();
        float time = timer.Elapsed();
        printf("Processing time of all device streams: %f ms\n\n", time);

        // TODO: Destroy device streams
		for (int i = 0; i < nStreams; i++)
		{
			cudaStreamDestroy(streams[i]);
		}

        // TODO: Free device memory regions
		for (int i = 0; i < nStreams; i++)
		{
			cudaFree(d_vectors[i]);
		}
	}
}

int main(int argc, char ** argv)
{
    int n; 
    int *in1, *in2; 
    int *out, *correctOut;

    // Input data into n
    n = (1 << 24) + 1;
    printf("n =  %d\n\n", n);

    // Allocate memories for in1, in2, out
    size_t nBytes = n * sizeof(int);
    CHECK(cudaMallocHost(&in1, nBytes));
    CHECK(cudaMallocHost(&in2, nBytes));
    CHECK(cudaMallocHost(&out, nBytes));
    correctOut = (int *)malloc(nBytes);

    // Input data into in1, in2
    for (int i = 0; i < n; i++)
    {
    	in1[i] = rand() & 0xff; // Random int in [0, 255]
    	in2[i] = rand() & 0xff; // Random int in [0, 255]
    }

    // Add in1 & in2 on host
    addVec(in1, in2, n, correctOut);

    // Add in1 & in2 on device
	dim3 blockSize(512); // Default
    int nStreams = 1; // Default
	if (argc >= 2)
	{
		blockSize.x = atoi(argv[1]);
        if (argc >= 3)
        {
            nStreams = atoi(argv[2]);
        }
	} 
    addVec(in1, in2, n, out, true, blockSize, nStreams);

    // Check correctness
    for (int i = 0; i < n; i++)
    {
    	if (out[i] != correctOut[i])
    	{
    		printf("INCORRECT :(\n");
    		return 1;
    	}
    }
    printf("CORRECT :)\n");
    
    CHECK(cudaFreeHost(in1));
    CHECK(cudaFreeHost(in2));
    CHECK(cudaFreeHost(out));    
    free(correctOut);
}
