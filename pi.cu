#include <iostream>
#include <vector>

using namespace std;

// Create a kernel to estimate pi
__global__ 
void count_samples_in_circles(float* d_randNumsX, float* d_randNumsY, int* d_countInBlocks, int num_blocks, int nsamples) 
{

  __shared__ int shared_blocks[500];

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * num_blocks;

  // Iterates through
  int inCircle = 0;  
  for (int i = index; i < nsamples; i+= stride) {    
    float xValue = d_randNumsX[i];    
    float yValue = d_randNumsY[i];

    if (xValue*xValue + yValue*yValue <= 1.0f) {
      inCircle++;    
    }  
  }
  
  shared_blocks[threadIdx.x] = inCircle;

  __syncthreads();

  // Pick thread 0 for each block to collect all points from each Thread.
  if (threadIdx.x == 0) 
  {    
    int totalInCircleForABlock = 0;    
    for (int j = 0; j < blockDim.x; j++) 
    {      
      totalInCircleForABlock += shared_blocks[j];    
    }
    d_countInBlocks[blockIdx.x] = totalInCircleForABlock;  
  }
}

int nsamples = 1e8;

int main(void) {   
  // allocate space to hold random values    
  vector<float> h_randNumsX(nsamples);    
  vector<float> h_randNumsY(nsamples);
  srand(time(NULL)); // seed with system clock
    
  //Initialize vector with random values    
  for (int i = 0; i < h_randNumsX.size(); ++i) 
  {        
    h_randNumsX[i] = float(rand()) / RAND_MAX;        
    h_randNumsY[i] = float(rand()) / RAND_MAX;    
  }
  
  // Send random values to the GPU    
  size_t size = nsamples * sizeof(float);    
  float* d_randNumsX;    
  float* d_randNumsY;    
  
  cudaMalloc(&d_randNumsX, size);  
  cudaMalloc(&d_randNumsY, size);
  cudaMemcpy(d_randNumsX, &h_randNumsX.front(), size, cudaMemcpyHostToDevice);    
  cudaMemcpy(d_randNumsY, &h_randNumsY.front(), size, cudaMemcpyHostToDevice);
  
  // Launch kernel to count samples that fell inside unit circle    
  int threadsPerBlock = 500;
  int num_blocks = nsamples / (1000 * threadsPerBlock);
  size_t countBlocks = num_blocks * sizeof(int);
 
  int* d_countInBlocks;
  cudaMalloc(&d_countInBlocks, countBlocks);

  // CALL KERNEL  
  count_samples_in_circles<<<num_blocks, threadsPerBlock>>>(d_randNumsX, d_randNumsY, d_countInBlocks, num_blocks, nsamples);
  if ( cudaSuccess != cudaGetLastError() )
    cout << "Error!\n";

  // Return back the vector from device to host
  int* h_countInBlocks = new int[num_blocks];
  cudaMemcpy(h_countInBlocks, d_countInBlocks, countBlocks, cudaMemcpyDeviceToHost);

  int nsamples_in_circle = 0;
  for (int i = 0 ; i < num_blocks; i++) {
    //cout << "Value in block " + i << " is " << h_countInBlocks[i] << endl;
    nsamples_in_circle = nsamples_in_circle + h_countInBlocks[i];
  }

  cudaFree(d_randNumsX);
  cudaFree(d_randNumsY);
  cudaFree(d_countInBlocks);

  // fraction that fell within (quarter) of unit circle
  float estimatedValue = 4.0 * float(nsamples_in_circle) / nsamples;

  cout << "Estimated Value: " << estimatedValue << endl;
}

