# python_cuda_numba
python uses cuda via numba template

# setup environment
## using yml
```
conda env create -f=/path/to/env.yml
```

## or, step by step
```
conda create -n cuda python=3.6.7
source activate cuda
conda install cudatoolkit # depending on driver, you may want install other version. see below how
conda install six numba
python sum.py
```


## figure out your cudatoolkit version options

### why?
You experience below when running the example
```
CUDA_ERROR_NO_BINARY_FOR_GPU ptxas application ptx input, line 9; fatal : Unsupported .version 6.2; current version is '5.0'
```

### how?
you install other cudatoolkit version
```
>conda search cudatoolkit
Fetching package metadata .............
cudatoolkit                  5.5rc1                       p0  defaults
                             5.5.1                        p0  defaults
                             6.0                          p0  defaults
                             7.0                           1  defaults
                             7.5                           0  defaults
                             7.5                           2  defaults
                             8.0                           1  defaults
                             8.0                           3  defaults
                             9.0                  h13b8566_0  defaults
                             9.2                           0  defaults

>conda install cudatoolkit=8.0 # for example
```

## profiling
```
>nfprof python sum.py
[2 2 2 2 2]
cpu elapsed 0.862177848815918 sec
==10396== NVPROF is profiling process 10396, command: python sum.py
[2 2 2 2 2]
gpu elapsed 1.4163212776184082 sec
==10396== Profiling application: python sum.py
==10396== Profiling result:
Time(%)      Time     Calls       Avg       Min       Max  Name
 57.45%  664.60ms         1  664.60ms  664.60ms  664.60ms  [CUDA memcpy DtoH]
 40.64%  470.04ms         2  235.02ms  233.04ms  237.00ms  [CUDA memcpy HtoD]
  1.91%  22.103ms         1  22.103ms  22.103ms  22.103ms  cudapy::__main__::__vect
orized_vec_sum$242(Array<short, int=1, A, mutable, aligned>, Array<short, int=1, A,
 mutable, aligned>, Array<short, int=1, A, mutable, aligned>)
```
gpu uses only 22ms for the computing, though 1.13s for transfering data between device and host.
cpu uses 0.86s
