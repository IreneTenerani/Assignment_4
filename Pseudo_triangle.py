import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline

class Line:
    def __init__(self, slope=1., intercept=0., minx=0., maxx=0.):
        self._slope = slope
        self._intercept = intercept
        self._minx = minx
        self._maxx= maxx

    def __str__(self):
        return 'y= {} x + {}'.format(self._slope, self._intercept)

    def line (self, x=0.):
        return self._slope * x + self._intercept

    def draw(self):
        a=np.linspace(self._minx, self._maxx, 100)
        plt.plot(a, self.line(a), 'r-')
        plt.show()


class Distribution:
    "Class representing a triangle"
    def __init__(self, x, y):
        self._x=np.array(x)
        self._y=np.array(y)

    def interpolate(self, grade=0.):
        spl= InterpolatedUnivariateSpline(self._x, self._y, k=grade)
        integral= spl.integral(0., 10.)
        print('integrale sotto la curva {}'.format(integral))


    def draw(self, grade=0.):
        a=np.linspace(0, 10, 100)
        spl=InterpolatedUnivariateSpline(self._x, self._y, k=grade)
        plt.plot(self._x, self._y, 'bo')
        plt.plot(a, spl(a), 'r-')
        plt.show()

    def __str__(self):
        return str((tuple(self._x), tuple(self._y)))



#Sampling the Distribution--> need to put this into the class distribution
l1=Line(1., 0., 0., 10.,)
x=np.array([1, 1.5, 3, 4.3, 6, 7, 8.1, 9.4])
y=l1.line(x)

triangle=Distribution(x,y)

#output
print(x)
print(y)
print(triangle)


print(l1)
l1.draw()
triangle.interpolate(1)
triangle.draw(1)
