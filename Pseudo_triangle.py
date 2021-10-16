import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.stats import norm
from scipy.optimize import curve_fit


class ProbabilityDensityFunction(InterpolatedUnivariateSpline):

    def __init__(self, x, y):
        self._x=np.array(x)
        self._y=np.array(y)

        InterpolatedUnivariateSpline.__init__(self, x, y)
        self.spl= InterpolatedUnivariateSpline(self._x, self._y)

        self.cdf=self.spl.antiderivative()
        ycdf = np.array([(self.cdf(xcdf)-self.cdf(self._x[0])) for xcdf in self._x])

        xppf, ippf = np.unique(ycdf, return_index=True)
        yppf=self._x[ippf]

        self.ppf = InterpolatedUnivariateSpline(xppf, yppf)



    def prob(self, x1, x2):
        """Return the probability for the random variable to be included
        between x1 and x2.
        """
        return self.cdf(x2) - self.cdf(x1)

    def rnd(self, size=1000):
        return self.ppf(np.random.uniform(size=size))


    def draw(self):
        a=np.linspace(0, 1, 100)
        plt.plot(self._x, self._y, 'bo')
        plt.plot(a, self.spl(a), 'r-')
        plt.plot(a, self.ppf(a), 'y')
        plt.plot(a, self.cdf(a), 'b')
        plt.show()

    def __str__(self):
        return str((tuple(self._x), tuple(self._y)))


def test_triangular():
    """Unit test with a triangular distribution.
    """
    x = np.linspace(0., 1., 101)
    y = 2. * x
    pdf = ProbabilityDensityFunction(x, y)
    a = np.array([0.2, 0.6])
    print(pdf(a))

    plt.figure('pdf')
    plt.plot(x, pdf(x))
    plt.xlabel('x')
    plt.ylabel('pdf(x)')

    plt.figure('cdf')
    plt.plot(x, pdf.cdf(x))
    plt.xlabel('x')
    plt.ylabel('cdf(x)')

    plt.figure('ppf')
    q = np.linspace(0., 1., 250)
    plt.plot(q, pdf.ppf(q))
    plt.xlabel('q')
    plt.ylabel('ppf(q)')

    plt.figure('Sampling')
    rnd = pdf.rnd(1000000)

    ydata, edges, _ = plt.hist(rnd, bins=200)
    xdata = 0.5 * (edges[1:] + edges[:-1])

    def f(x, m, q):
        return m * x + q

    popt, pcov = curve_fit(f, xdata, ydata)
    print(popt)
    print(np.sqrt(pcov.diagonal()))

    _x = np.linspace(0, 1, 100)
    _y = f(_x, *popt)
    plt.plot(_x, _y)

    chi2 = sum(((ydata - f(xdata, *popt)) / np.sqrt(ydata))**2.)
    nu = len(ydata) - len(popt)
    sigma = np.sqrt(2 * nu)
    print(chi2, nu, sigma)


def test_triangular_personal():

    def Retta(x, slope=1., intercept=0.):
        return slope * x + intercept

    x=np.linspace(0., 1., 100)
    y=Retta(x, 2., 0.)

    triangle=ProbabilityDensityFunction(x,y)

    plt.figure('Sampling')
    plt.hist(triangle.rnd(1000000), bins=200)

    plt.figure('Graphs')
    a=np.linspace(0, 1, 100)
    plt.plot(triangle._x, triangle._y, 'bo')
    plt.plot(a, triangle.spl(a), 'r-')
    plt.plot(a, triangle.ppf(a), 'y')
    plt.plot(a, triangle.cdf(a), 'b')


    probabilita=triangle.prob(0., 1.0)
    print('area sotto triangolo è {}'.format(triangle.spl.integral(0. ,1.)))
    print('La probabilità è {}'.format(probabilita))
    print('area della ppf è {}'.format(triangle.ppf.integral(0. ,1.)))
    print('area della cdf è {}'.format(triangle.cdf.integral(0. ,1.)))
    print('alcuni valori della ppf {}'.format(triangle.ppf([0., 0.1, 0.5, 1.])))

def test_gauss(mu=0., sigma=1., support=10., num_points=500):
    """Unit test with a gaussian distribution.
    """
    x = np.linspace(-support * sigma + mu, support * sigma + mu, num_points)
    y = norm.pdf(x, mu, sigma)
    pdf = ProbabilityDensityFunction(x, y)

    plt.figure('pdf')
    plt.plot(x, pdf(x))
    plt.xlabel('x')
    plt.ylabel('pdf(x)')

    plt.figure('cdf')
    plt.plot(x, pdf.cdf(x))
    plt.xlabel('x')
    plt.ylabel('cdf(x)')

    plt.figure('ppf')
    q = np.linspace(0., 1., 1000)
    plt.plot(q, pdf.ppf(q))
    plt.xlabel('q')
    plt.ylabel('ppf(q)')

    plt.figure('Sampling')
    rnd = pdf.rnd(1000000)

    ydata, edges, _ = plt.hist(rnd, bins=200)
    xdata = 0.5 * (edges[1:] + edges[:-1])

    def f(x, C, mu, sigma):
        return C * norm.pdf(x, mu, sigma)

    popt, pcov = curve_fit(f, xdata, ydata)
    print(popt)
    print(np.sqrt(pcov.diagonal()))
    _x = np.linspace(-10, 10, 500)
    _y = f(_x, *popt)
    plt.plot(_x, _y)

    mask = ydata > 0
    chi2 = sum(((ydata[mask] - f(xdata[mask], *popt)) / np.sqrt(ydata[mask]))**2.)
    nu = mask.sum() - len(popt)
    sigma = np.sqrt(2 * nu)
    print(chi2, nu, sigma)


if __name__ == '__main__':
    test_triangular()
    plt.show()
    test_triangular_personal()
    plt.show()
    test_gauss()
    plt.show()
