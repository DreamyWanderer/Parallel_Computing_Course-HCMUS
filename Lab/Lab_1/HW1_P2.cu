#include <stdio.h>
#include <stdint.h>

constexpr int FILTER_WIDTH = 9;
const int8_t idx[FILTER_WIDTH * FILTER_WIDTH] = 
						{-4, -4, -4, -4, -4, -4, -4, -4, -4,
					   -3, -3, -3, -3, -3, -3, -3, -3, -3,
					   -2, -2, -2, -2, -2, -2, -2, -2, -2,
					   -1, -1, -1, -1, -1, -1, -1, -1, -1,
					   0, 0, 0, 0, 0, 0, 0, 0, 0,
					   1, 1, 1, 1, 1, 1, 1, 1, 1,
					   2, 2, 2, 2, 2, 2, 2, 2, 2,
					   3, 3, 3, 3, 3, 3, 3, 3, 3,
					   4, 4, 4, 4, 4, 4, 4, 4, 4};
const int8_t idy[FILTER_WIDTH * FILTER_WIDTH] = 
						{-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4,
						-4, -3, -2, -1, 0, 1, 2, 3, 4};

__constant__ int8_t d_idx[FILTER_WIDTH * FILTER_WIDTH],
					d_idy[FILTER_WIDTH * FILTER_WIDTH];

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

void print3x3PixelSquare(uchar3 *pixel, int r, int c, int width, int height)
{
	const int8_t idx[9] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
	const int8_t idy[9] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
	for (int i = 0; i < 9; i++)
	{
		int tmpR = r + idx[i], tmpC = c + idy[i];
		int tmpId = tmpR * width + tmpC;
		printf("(%d %d %d) ", pixel[tmpId].x, pixel[tmpId].y, pixel[tmpId].z);
		if ( (i + 1) % 3 == 0 ) printf("\n");
	}
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

void readPnm(char * fileName, 
		int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

__global__ void blurImgKernel(uchar3 * inPixels, int width, int height, 
		float * filter, int filterWidth, uchar3 * outPixels)
{
	// TODO
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	if (r >= height && c >= width) return;

	float redSum = 0, greenSum = 0, blueSum = 0;
	int tmpR, tmpC;

	// Compute new value for pixel at (r, c) from surrounding pixels
	for (int i = 0; i < filterWidth * filterWidth; i++)
	{
		// Get index of a pixel surrounds the computing pixel
		tmpR = r + d_idx[i]; 
		tmpC = c + d_idy[i];
		// If the pixel is outside of valid range: get the nearest proper index
		tmpR = (tmpR < 0) ? (0) : tmpR;
		tmpR = (tmpR >= height) ? (height - 1) : tmpR;
		tmpC = (tmpC < 0) ? (0) : tmpC;
		tmpC = (tmpC >= width) ? (width - 1) : tmpC;
		
		redSum += (inPixels[ tmpR * width + tmpC].x * filter[0]);
		greenSum += (inPixels[ tmpR * width + tmpC].y * filter[0]);
		blueSum += (inPixels[ tmpR * width + tmpC].z * filter[0]);
	}

	int tmpId = r * width + c;
	outPixels[tmpId].x = redSum; outPixels[tmpId].y = greenSum; outPixels[tmpId].z = blueSum;
}

void blurImg(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
		uchar3 * outPixels,
		bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		// TODO
		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width; c++)
			{
				float redSum = 0, greenSum = 0, blueSum = 0;
				int tmpR, tmpC;

				// Compute new value for pixel at (r, c) from surrounding pixels
				for (int i = 0; i < filterWidth * filterWidth; i++)
				{
					// Get index of a pixel surrounds the computing pixel
					tmpR = r + idx[i]; 
					tmpC = c + idy[i];
					// If the pixel is outside of valid range: get the nearest proper index
					tmpR = (tmpR < 0) ? (0) : tmpR;
					tmpR = (tmpR >= height) ? (height - 1) : tmpR;
					tmpC = (tmpC < 0) ? (0) : tmpC;
					tmpC = (tmpC >= width) ? (width - 1) : tmpC;
					
					redSum += (inPixels[ tmpR * width + tmpC].x * filter[0]);
					greenSum += (inPixels[ tmpR * width + tmpC].y * filter[0]);
					blueSum += (inPixels[ tmpR * width + tmpC].z * filter[0]);
				}

				int tmpId = r * width + c;
				outPixels[tmpId].x = redSum; outPixels[tmpId].y = greenSum; outPixels[tmpId].z = blueSum;
				//printf("%d %d %d\n", outPixels[tmpId].x, outPixels[tmpId].y, outPixels[tmpId].z);
			}
		}

	}
	else // Use device
	{
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);

		// TODO
		uchar3 *d_InPixel, *d_Outpixel;
		float *d_filter;
		const int sizeInPixel = height * width * sizeof(uchar3) * 3;
		const int sizeOutPixel = height * width * sizeof(uchar3);
		const int sizeFilter = filterWidth * filterWidth * sizeof(float);
		const int sizeIdx = filterWidth * filterWidth * sizeof(int8_t), sizeIdy = sizeIdx;

		CHECK( cudaMalloc(&d_InPixel, sizeInPixel));
		CHECK( cudaMalloc(&d_Outpixel, sizeInPixel));
		CHECK( cudaMalloc(&d_filter, sizeFilter));

		CHECK( cudaMemcpy(d_InPixel, inPixels, sizeInPixel, cudaMemcpyHostToDevice));
		CHECK( cudaMemcpy(d_filter, filter, sizeFilter, cudaMemcpyHostToDevice));
		CHECK( cudaMemcpyToSymbol(d_idx, idx, sizeIdx));
		CHECK( cudaMemcpyToSymbol(d_idy, idy, sizeIdx));

		dim3 gridSize( ceil(width/blockSize.x), ceil(height/blockSize.y));
		blurImgKernel<<<gridSize, blockSize>>>(d_InPixel, width, height, d_filter, filterWidth, d_Outpixel);
		CHECK( cudaGetLastError());
		CHECK( cudaMemcpy(outPixels, d_Outpixel, sizeInPixel, cudaMemcpyDeviceToHost));
		printf("%d\n", sizeIdy);


		CHECK( cudaFree(d_InPixel));
		CHECK( cudaFree(d_Outpixel));
		CHECK( cudaFree(d_filter));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n", 
    		useDevice == true? "use device" : "use host", time);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, char ** argv)
{
	if (argc != 4 && argc != 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);
	print3x3PixelSquare(inPixels, 1, 1, width, height);

	// Read correct output image file
	int correctWidth, correctHeight;
	uchar3 * correctOutPixels;
	readPnm(argv[3], correctWidth, correctHeight, correctOutPixels);
	if (correctWidth != width || correctHeight != height)
	{
		printf("The shape of the correct output image is invalid\n");
		return EXIT_FAILURE;
	}
	printf("Image size (width x height): %i x %i\n\n", width, height);
	print3x3PixelSquare(correctOutPixels, 1, 1, width, height);

	// Set up a simple filter with blurring effect 
	int filterWidth = 9;
	float * filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	for (int filterR = 0; filterR < filterWidth; filterR++)
	{
		for (int filterC = 0; filterC < filterWidth; filterC++)
		{
			filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
		}
	}
	//for (int i = 0; i < filterWidth * filterWidth; i++) printf("%f ", filter[i]);

	// Blur input image using host
	uchar3 * hostOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3)); 
	blurImg(inPixels, width, height, filter, filterWidth, hostOutPixels);
	print3x3PixelSquare(hostOutPixels, 1, 1, width, height);
	
	// Compute mean absolute error between host result and correct result
	float hostErr = computeError(hostOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", hostErr);

	// Blur input image using device
	uchar3 * deviceOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}  
	blurImg(inPixels, width, height, filter, filterWidth, deviceOutPixels, true, blockSize);

	// Compute mean absolute error between device result and correct result
	float deviceErr = computeError(deviceOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", deviceErr);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(correctOutPixels);
	free(hostOutPixels);
	free(deviceOutPixels);
	free(filter);
}
