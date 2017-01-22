from pylab import *
import numpy as np
import scipy as sp
from scipy.ndimage import imread
from scipy.misc import imsave
from scipy.signal import convolve2d as conv

def anisotropicGauss1(sigmax, sigmay, shape=(3,3)):
    m,n = shape
    kernel = np.zeros((m, n))

    for x in range(m):
        for y in range(n):
            dist = hypot((x-(m/2)), (y-(n/2))) #Euclidean distance
            value = (1.0/2*pi*sigmax*sigmay)*exp(-(dist**2)/(2*sigmax*sigmay))
            kernel[x,y] = value
    return kernel/np.sum(kernel)

def anisotropicGauss2(sigmax, sigmay, shape=(3,3)):
    m,n = shape
    kernel = np.zeros((m, n))

    for x in range(m):
        for y in range(n):
            distx, disty = x-(m/2), y-(n/2)
            value = (1.0/2*pi*sigmax*sigmay)*exp(((distx*distx/(-2*sigmax*sigmax)) +
                                                  (disty*disty/(-2*sigmay*sigmay))))
            kernel[x,y] = value
    return kernel/np.sum(kernel)

def anisotropicGauss(sigmax, sigmay, size=(3,3)):
    """
    kernel size = 3*sigma for gaussian filters (make a decently sized kernel)
    """
    m,n = tuple(ss*sigmax+sigmay for ss in size)
    kernel = np.zeros((m, n))

    for x in range(m):
        for y in range(n):
            distx, disty = x-(m/2), y-(n/2) #centre pixels
            value = (1.0/sqrt(2*pi)*sigmax)*exp((distx**2/(-1.0*(sigmax**2))))*\
                (1.0/sqrt(2*pi)*sigmay)*exp((disty**2/(-1.0*(sigmay**2))))
            kernel[x,y] = value
            
    return kernel/np.sum(kernel)

def matlab_gauss2D(sigmax, sigmay, shape=(3,3)):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigmax*sigmay) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

if __name__ == "__main__":
    image = imread("cat.jpg", mode='L')
    kernel = anisotropicGauss(15, 2) 
    result = conv(image, kernel, mode='same', boundary = 'fill', fillvalue = 0)
    imsave("blurred_cat.jpg", result)