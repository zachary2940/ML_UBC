import numpy as np
#write code that computes the gradient of the function
# x is a 1-d array
def example(x):
    return np.sum(x**2)

def example_grad(x):
    return 2*x

def foo(x):
    result = 1
    λ = 4 # this is here to make sure you're using Python 3
    for x_i in x: # x is random vector
        result += x_i**λ
    return result # 1+x^4+y^4+...
'''
Forward finite difference
         f(xk[i] + epsilon[i]) - f(xk[i])
f'[i] = ---------------------------------
                    epsilon[i]

h = 0.00000000001
grad = numpy.zeros((len(x),), float) # creating array of zeros
for i,x_i in enumerate(x):
    top = fun(x_i + h) - fun(x_i)
    bottom = h
    slope = top / bottom    # Returns the slope to the third decimal
    grad[i] = slope
return float("%.3f" % grad)
'''

def foo_grad(x): # getting specific gradient vector so for function foo
    grad = np.zeros((len(x),), float)
    for i,x_i in enumerate(x):
        grad[i] = 4*x_i**3
    return grad

def bar(x): #Return the product of array elements over a given axis. ie [1,2,3]=6
    return np.prod(x) #xyz

def bar_grad(x):
    grad = np.zeros((len(x),), float)
    for i,x_i in enumerate(x):
        grad[i] = np.prod(x)/x_i
    return grad

