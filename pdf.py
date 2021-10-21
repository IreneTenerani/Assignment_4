import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from scipy.optimize import curve_fit
#pylint: disable=invalid-name

class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    '''
    ProbabilityDensityFunction class that is capable of throwing
    preudo-random number with an arbitrary distribution.


    Parameters
    __________

    x : (N,) array_like
        1-D array of independent input data. Must be increasing .

    y : (N,) array_like
        1-D array of dependent input data, of the same length as `x`.

    '''

    def __init__(self, x, y):

        self._x=np.array(x)
        self._y=np.array(y)

        #Building a spline representing the pdf function, passing through the input data .
        InterpolatedUnivariateSpline.__init__(self, x, y)
        self.spl= InterpolatedUnivariateSpline(self._x, self._y)

        #Building the cdf function and sapling it in x data to obtain ycdf values .
        self.cdf=self.spl.antiderivative()
        ycdf = np.array([(self.cdf(xcdf)-self.cdf(self._x[0])) for xcdf in self._x])

        #Sorting the ycdf array and storing it in xppf .
        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf=self._x[ippf]

        #Building the ppf function .
        self.ppf = InterpolatedUnivariateSpline(xppf, yppf)


    def prob(self, x1, x2):
        '''Return the probability for the random variable to be included
        between x1 and x2.

        Parameters

        __________

        x1 : float, lower edge of the intervall

        x2 : floar, higher edge of the intervall

        Returns

        _______

        prob : float, the probability for the random variable to be included between x1 and x2.
        '''
        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, size=1000):
        '''Return random numbers distributed accordingly to the pdf function .

        Parameters

        __________

        size : integer, lenght of the array of random numbers generated .

        Returns

        _______

        rnd : array of floats, random numbers distributed accordingly to the pdf function

        '''
        return self.ppf(np.random.uniform(size=size))

    def __str__(self):
        return str((tuple(self._x), tuple(self._y)))
