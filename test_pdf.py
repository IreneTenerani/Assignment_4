import unittest
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from scipy.optimize import curve_fit

from pdf import ProbabilityDensityFunction

'''
def absolute_data_folder_path():
    """Ugly hack for finding the path to the data folder. Works only if the
    current working directory is inside the cmepda package (any of the
    subfolder will do)"""
    import os
    cwd = os.getcwd()
    idx = cwd.rfind('Assegnamento_basic_4')
    if idx == -1:
        raise IOError('Could not locate the data folder. The test must be run\
                      from inside the Assegnamento_basic_4 package.')
    else:
        Assegnamento_basic_4_dir = cwd[:idx + len('Assegnamento_basic_4')]
    return os.path.(Assegnamento_basic_4_dir)
'''


class TestPdf(unittest.TestCase):

    def setUp(self,m=2.0,  mu=0., sigma=1., support=10., num_points=500):


        self.xretta=np.linspace(0., 1., 100)
        self.yretta=m*self.xretta

        self.support=support
        self.xgauss=np.linspace(-support * sigma + mu, support * sigma + mu, num_points)
        self.ygauss=norm.pdf(self.xgauss, mu, sigma)

    def test_probability(self):

        triangle=ProbabilityDensityFunction(self.xretta,self.yretta)
        gauss=ProbabilityDensityFunction(self.xgauss,self.ygauss)

        self.assertAlmostEqual(triangle.prob(0., 1.0), 1.0)
        self.assertAlmostEqual(gauss.prob(-self.support, self.support), 1.0)


    def test_plotting(self, draw=True):
        """ Test plotting."""
        triangle=ProbabilityDensityFunction(self.xretta,self.yretta)
        # Draw the plots into our figure
        plt.figure('test_plot_retta')
        a=np.linspace(0, 1, 100)
        plt.plot(triangle._x, triangle._y, 'bo')
        plt.plot(a, triangle.spl(a), 'r-')
        plt.plot(a, triangle.ppf(a), 'y')
        plt.plot(a, triangle.cdf(a), 'b')
        if draw:
            plt.show()

if __name__ == '__main__':
    unittest.main()
