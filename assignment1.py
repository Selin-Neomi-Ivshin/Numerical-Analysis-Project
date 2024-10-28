"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random



class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time.
        The assignment will be tested on variety of different functions with
        large n values.

        Interpolation error will be measured as the average absolute error at
        2*n random points between a and b. See test_with_poly() below.

        Note: It is forbidden to call f more than n times.

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.**

        Note: sometimes you can get very accurate solutions with only few points,
        significantly less than n.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        def TDMA_Solver(a, b, c, d): #Thomas algorithm
            n = len(d)
            w = np.zeros(n - 1, float)
            g = np.zeros(n, float)
            p = np.zeros(n, float)

            w[0] = c[0] / b[0]
            g[0] = d[0] / b[0]

            for i in range(1, n - 1):
                w[i] = c[i] / (b[i] - a[i - 1] * w[i - 1])
            for i in range(1, n):
                g[i] = (d[i] - a[i - 1] * g[i - 1]) / (b[i] - a[i - 1] * w[i - 1])
            p[n - 1] = g[n - 1]
            for i in range(n - 1, 0, -1):
                p[i - 1] = g[i - 1] - w[i - 1] * p[i]
            return p

        def bezier3(P1, P2, P3, P4): #bezier algorithm
            M = np.array(
                [[-1, +3, -3, +1],
                 [+3, -6, +3, 0],
                 [-3, +3, 0, 0],
                 [+1, 0, 0, 0]],
                dtype=np.float32
            )
            P = np.array([P1, P2, P3, P4], dtype=np.float32)

            def f(t):
                T = np.array([t ** 3, t ** 2, t, 1], dtype=np.float32)
                return T.dot(M).dot(P)

            return f


        if n == 1: #Extreme case
            func = lambda num: f((a+b)/2)
            return func

        x_point_value = np.linspace(a, b, n) #getting the x value of the point

        y_point_value = []
        for i in x_point_value: #getting the y value of each point
            y_point_value.append(f(i))

        points_lst = np.array(list(zip(x_point_value, y_point_value))) #the function's points

        number_of_splines = n - 1


        scalar_mul_res = [] #crating the result of matrix*vector_a
        scalar_mul_res.append(points_lst[0] + (2 * points_lst[1]))
        for i in range(1, number_of_splines - 1):
            scalar_mul_res.append(4 * points_lst[i] + 2 * points_lst[i + 1])
        scalar_mul_res.append((8 * points_lst[number_of_splines - 1]) + points_lst[number_of_splines]) #the last vector is speacial

        def filling_diagonal_lines(matrix, num): # function for filling the diagonal of the matrix
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    if i == j:
                        matrix[i][j] = num

        matrix = np.zeros((number_of_splines, number_of_splines))  # creating "empty" matrix


        filling_diagonal_lines(matrix, 4) #filling the diagonal with 4's
        filling_diagonal_lines(matrix[1:, :], 1) #slicing the first row of the matrix and filling the new diagonal with 1's
        filling_diagonal_lines(matrix[:, 1:], 1) #slicing the first column of the matrix and filling the new diagonal with 1's

        matrix[0, 0] = 2 #changing 4 by 2
        matrix[number_of_splines - 1, number_of_splines - 1] = 7 #changing the last index in the matrix
        matrix[number_of_splines - 1, number_of_splines - 2] = 2 #changing the before last index in the matrix

        lower_diagonal= np.diag(matrix, k=-1) #extracting the lower diagonal from the matrix
        diagonal = np.diag(matrix, k=0)#extracting the diagonal from the matrix
        upper_diagonal = np.diag(matrix, k=1)#extracting the upper diagonal from the matrix

        x_of_scalar_mul_res = [] #list to hold the x points of the scalar_mul_res
        for point in scalar_mul_res:
            x_of_scalar_mul_res.append(point[0])

        y_of_scalar_mul_res = [] #list to hold the y points of the scalar_mul_res
        for point in scalar_mul_res:
            y_of_scalar_mul_res.append(point[1])

        x_of_vec_a = TDMA_Solver(lower_diagonal, diagonal, upper_diagonal, x_of_scalar_mul_res) #calling TMDA function
        y_of_vec_a = TDMA_Solver(lower_diagonal, diagonal, upper_diagonal, y_of_scalar_mul_res) #calling TMDA function

        vector_a = np.array(list(zip(x_of_vec_a, y_of_vec_a))) #creating vector a

        vector_b = [] #creating vector b by using the equation we learned in the lecture
        for i in range(0, number_of_splines - 1):
            vector_b.append((2 * points_lst[i + 1]) - vector_a[i + 1])
        vector_b.append((vector_a[number_of_splines - 1] + points_lst[number_of_splines])/2) #the last vector is special

        dictionary = {}
        for i in range(number_of_splines): #finding the result for bezier algorithm
            dictionary[(points_lst[i][0], points_lst[i + 1][0])] = bezier3(points_lst[i], vector_a[i], vector_b[i], points_lst[i + 1])


        def approx_func(num):
            first_point = (num - a) / ((b - a) / (n - 1)) #find x location
            first_x = (np.floor(first_point) * ((b - a) / (n - 1))) + a
            second_x = (np.ceil(first_point) * ((b - a) / (n - 1))) + a
            if first_x == second_x:
                t = 0
                second_x += ((b - a) / (n - 1))
            else:
                t = (num - first_x) / (second_x - first_x) #Finding the relative position of x on the appropriate curve
            return dictionary[(first_x, second_x)](t)[1] #return the y


        return approx_func









##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
