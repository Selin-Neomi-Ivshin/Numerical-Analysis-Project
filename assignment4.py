"""
In this assignment you should fit a model function of your choice to data
that you sample from a given function.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you take an iterative approach and know that
your iterations may take more than 1-2 seconds break out of any optimization
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools
for solving this assignment.

"""
import math
import time

import numpy as np
import timeit
import random

from numpy.ma import copy



class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

    def fit(self, f: callable, a: float, b: float, d:int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        f : callable.
            matrix_h function which returns an approximate (noisy) Y value given X.
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        a function:float->float that fits f between a and b
        """
        start_sample = time.time() #start counting time
        samp = f(a) #taking a sample from f
        end_sample = time.time() #end counting time
        time_to_samp = end_sample - start_sample #how much time it takes to take a sample
        if time_to_samp <= 0.0001:
            samples = 10000 #big number of samples
        else:
            samples = int(abs(((maxtime - (maxtime / 5)) / time_to_samp) - 1)) #small number of samples


        list_of_x_points = np.linspace(a, b, samples)  # creating a list of the x's if the fitting range
        ys = [samp] + [f(point) for point in list_of_x_points[1:]] #adding the first sample to the other samples
        list_y_points = np.array(ys)  # creating a list of the y's if the fitting range



        upper = np.empty(4)  # creating an empty array with 4 places
        list_index = 0
        for i in range(3, -1, -1):
            upper[list_index] = i  # filling the upper array with numbers from 0 to 3. This array represent the powers
            list_index += 1


        matrix = np.array([[-1, 3, -3, 1],
                           [3, -6, 3, 0],
                           [-3, 3, 0, 0],
                           [1, 0, 0, 0]], dtype=np.float32) #creating a matrix
        inversed_matrix = np.linalg.inv(matrix) #inversing the matrix


        index = 0
        A = np.empty((samples, 4)) #creating an empty matrix samplesX4
        for i in range(0, samples): #running all over the samples
            z = (list_of_x_points[i] - a) / (b - a) #normalize each x point to range [0,1]
            A[index] = np.power(z, upper) #put the normalized point in power - do it for all the upper array
            index += 1

        transposed_A = A.T #transpose A

        first = np.linalg.inv(np.dot(transposed_A, A)) #inverse the mul of the marix transposed_A and A
        second = np.dot(inversed_matrix, first) #mul the inversed matrix with the previous result
        third = np.dot(second, transposed_A) #mul the previous result with the transposed_A
        B = np.dot(third, list_y_points) #mul the previous result with the y points


        def approx_func(num):
            z = (num - a) / (b - a) #normalize the given point to range [0,1]
            AM = np.dot(np.array(np.power(z, upper)), matrix) #finding A by putting the normalized point in power and
            #mul it with matrix
            return np.float32(np.dot(AM, B)) #return the mul of AM and B

        return approx_func

##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1,1,1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #  f = DELAYED(7)(NOISY(0.01)(poly(1,1,1)))
    #
    #  ass4 = Assignment4()
    #  T = time.time()
    #  shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
    #  T = time.time() - T
    #  self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1,1,1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse=0
        for x in np.linspace(0,1,1000):
            self.assertNotEquals(f(x), nf(x))
            mse+= (f(x)-ff(x))**2
        mse = mse/1000
        print(mse)






if __name__ == "__main__":
    unittest.main()
