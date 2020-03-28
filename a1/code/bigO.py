import numpy as np

def func1(N): #O(N)
    for i in range(N):
        print("Hello!")

def func2(N): #O(N)
    x = np.zeros(N)
    x += 1000
    return x

def func3(N): #O(1)
    x = np.zeros(1000)
    x = x * N
    return x

def func4(N): #O(N^2)
    x = 0
    for i in range(N):
        for j in range(i,N):
            x += (i*j)
    return x