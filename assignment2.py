"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        func_for_finding_roots = lambda num: f1(num) - f2(num)  # the func that we try to find her roots

        def finding_roots(f, segment, maxerr):
            l_line = segment[0] #the left segment
            r_line = segment[1] #the right segment
            x = (l_line + r_line) / 2 #Our guess
            if abs(f(l_line)) <= maxerr: #if the x point on the left segment is the root we return it
                return l_line
            elif abs(f(r_line)) <= maxerr: #if the x point on the right segment is the root we return it
                return r_line
            else:
                while abs(func_for_finding_roots(x)) > maxerr: #while we don't find root
                    x = (l_line + r_line) / 2 #update the guess
                    if f(l_line) * f(x) < 0:
                        r_line = x #shrink
                    else:
                        l_line = x #shrink
                return x


        final = []  # saving the final result in an array
        segments = np.linspace(a, b, 1100)  # dividing the function into segments
        i = 0
        while (i + 1) < len(segments) - 1:  # while there are still segments in the function
            # If one point on the segment is above the x-axis and the other is under the x-axis
            if func_for_finding_roots(segments[i]) * func_for_finding_roots(segments[i + 1]) <= 0:
                # that means the function intersects the x-axis and we would like to find the root
                final.append(finding_roots(func_for_finding_roots, (segments[i], segments[i + 1]), maxerr))
                i = i + 1  # index promotion by 1
            else:  # If both points on the segment are above the x-axis or under the x-axis
                i = i + 1  # index promotion by 1

        # replace this line with your solution
        X = final
        return X


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
