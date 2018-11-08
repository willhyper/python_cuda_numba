#! /usr/bin/env python
import numpy as np
from numba import vectorize
import time

N = 1_000_000_000

@vectorize(['int16(int16, int16)'], target='cuda')
def vec_sum(a, b):
    return a + b
def sum(a, b):
    return a + b

def main():
    A = np.ones(N, dtype=np.int16)
    B = np.ones(N, dtype=np.int16)
    C = np.zeros(N, dtype=np.int16)
    
    start = time.time()
    C = sum(A, B)
    elapsed = time.time() - start

    print(C[:5])
    print('cpu elapsed', elapsed, 'sec')


    start = time.time()
    C = vec_sum(A, B)
    elapsed = time.time() - start

    print(C[:5])
    print('gpu elapsed', elapsed, 'sec')


if __name__ == '__main__':
    main()