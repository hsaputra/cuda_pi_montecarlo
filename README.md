# cuda_pi_montecarlo
CUDA code for Monte Carlo estimation of Pi (see https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/)

Using CUDA and GPGPU computing to caculate estimation of Pi using Monte Carlo with 100,000,000 rando points.

The code is using 500 threads per block and each process will process 1000 points.

Using the shared memory to collect points in a block.

Compile the code with `nvcc pi.cu -o pi` and then run the executable `pi`.

Make sure NVIDIA GPU is available with CUDA toolkit installed.
