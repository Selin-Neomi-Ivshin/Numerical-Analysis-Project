"""
In this assignment you should fit a model function of your choice to data
that you sample from a contour of given shape. Then you should calculate
the area of that shape.

The sampled data is very noisy so you should minimize the mean least squares
between the model you fit and the data points you sample.

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You
must make sure that the fitting function returns at most 5 seconds after the
allowed running time elapses. If you know that your iterations may take more
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment.
Note: !!!Despite previous note, using reflection to check for the parameters
of the sampled function is considered cheating!!! You are only allowed to
get (x,y) points from the given shape by calling sample().
"""
import numpy as np
import time
import random
from functionUtils import AbstractShape
from sklearn.cluster import KMeans


def angle_distance(xs, ys, first, second):
    first_value = xs #the x point
    first_value = first_value - first #substract from x point the minimum between the x points of the middle
    second_value = ys #the y point
    second_value = second_value - second #substract from y point the mean value of all y points of the middle

    interval1 = (first_value * first_value) #fist_value^2
    interval2 = (second_value * second_value) #second_value^2
    interval = interval1 + interval2 #distance^2
    zavit = np.arctan2(second_value, first_value) * ((360 * 2) / 4) / np.pi
    minus = -zavit #make the angle negative to run clockwise

    return (minus, interval)

class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, outlines: np.ndarray):
        self.outlines = outlines

    def contour(self, argument: int):
        return self.outlines[:argument]

    def sample(self):
        some_index = random.randrange(0, len(self.outlines), 1)
        sample1 = self.outlines[some_index]
        sample2 = self.outlines[some_index]
        return sample1, sample2

    def area(self):
        return Assignment5().area(contour=self.contour)


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        flag = True
        dots = 35 #using 35 points
        dictionary = {} #creating an empty dictionary

        i = 0
        while flag:
            dots1 = contour(dots) #use the dots to sample dots from the shape contour
            if i > 1: #if in the dictionary there are at least 2 values
                last_val = i - 1
                before_last = i - 2
                if abs((dictionary[last_val] - dictionary[before_last])/dictionary[last_val]) < maxerr: #find the relative error
                    return np.float32(dictionary[last_val]) #if the error is smaller than maxerr return the last value in the dictionary
            #shoelace method
            list_x_points = [sublist[0] for sublist in dots1]
            list_y_points = [sublist[1] for sublist in dots1]
            S1 = np.dot(list_x_points, np.roll(list_y_points, 1))
            S2 = np.dot(list_y_points, np.roll(list_x_points, 1))
            dictionary[i] = (0.5 * np.abs(S1-S2)) #update the dictionary
            i += 1
            dots += dots #enlarge the number of dots in order to be more accurate


    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """


        k_means = KMeans(n_clusters=25) #create 25 k-means

        dots = [] #create an empty list
        for i in range(2500):
            dots.append(sample()) #add to list 2500 samples

        k_means.fit(dots) #fit the k-means
        middle = k_means.cluster_centers_ #find the middle of each cluster of k-means

        dot_y = 0
        second_middle = [] #create an array of the y cordinate of the middle dots
        while dot_y < len(middle):
            second_middle.append(middle[dot_y][1])
            dot_y += 1

        dot_x = 0
        first_middle = []  #create an array of the x cordinate of the middle dots
        while dot_x < len(middle):
            first_middle.append(middle[dot_x][0])
            dot_x += 1

        #sort the middle dots using angle_distance function
        middle_sort = sorted(middle, key=lambda dot: angle_distance(dot[0], dot[1], np.min(first_middle), np.mean(second_middle)))
        return MyShape(np.array(np.float32(middle_sort)))





##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    # def test_delay(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        # def sample():
        #     time.sleep(7)
        #     return circ()
        #
        # ass5 = Assignment5()
        # T = time.time()
        # shape = ass5.fit_shape(sample=sample, maxtime=5)
        # T = time.time() - T
        # self.assertTrue(isinstance(shape, AbstractShape))
        # self.assertGreaterEqual(T, 5)

    # def test_circle_area(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=circ, maxtime=30)
    #     T = time.time() - T
    #     a = shape.area()
    #     self.assertLess(abs(a - np.pi), 0.01)
    #     self.assertLessEqual(T, 32)

    # def test_bezier_fit(self):
    #     circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
    #     ass5 = Assignment5()
    #     T = time.time()
    #     shape = ass5.fit_shape(sample=circ, maxtime=30)
    #     T = time.time() - T
    #     a = shape.area()
    #     self.assertLess(abs(a - np.pi), 0.01)
    #     self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
