# cuda_pi_montecarlo
CUDA code for Monte Carlo estimation of Pi (see https://www.geeksforgeeks.org/estimating-value-pi-usingmmonte-carlo/)

Using CUDA and GPGPU computing to caculate estimation of Pi using Monte Carlo with 100,000,000 random points.

The code is using 500 threads per block and each process will process 1000 points.

Using the shared memory to collect points in a block.

Compile the code with `nvcc pi.cu -o pi` and then run the executable `pi`.

Make sure NVIDIA GPU is available with CUDA toolkit installed.

```
nvcc pi.cu -o pi
./pi
Estimated Value: 3.14144
```

```
nvcc pi_timed.cu -o pi
./pi
Elapsed time Slow: 101.526
Elapsed time Coales: 6.80678
Estimated Value: 3.1417
```
