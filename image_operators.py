#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pierre Paleo <pierre.paleo@esrf.fr>
#  License: BSD 2-clause Simplified
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from __future__ import division
import numpy as np

def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient


def div(grad):
    '''
    Compute the divergence of a gradient
    Courtesy : E. Gouillart - https://github.com/emmanuelle/tomo-tv/
    '''
    res = np.zeros(grad.shape[1:])
    for d in range(grad.shape[0]):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        this_res[-1] -= this_grad[-2]
    return res


def psi(x, mu):
    '''
    Huber function needed to compute tv_smoothed
    '''
    res = np.abs(x)
    m = res < mu
    res[m] = x[m]**2/(2*mu) + mu/2
    return res


def tv_smoothed(x, mu):
    '''
    Moreau-Yosida approximation of Total Variation
    see Weiss, Blanc-Féraud, Aubert, "Efficient schemes for total variation minimization under constraints in image processing"
    '''
    g = gradient(x)
    g = np.sqrt(g[0]**2 + g[1]**2)
    return np.sum(psi(g, mu))

def grad_tv_smoothed(x, mu):
    '''
    Gradient of Moreau-Yosida approximation of Total Variation
    '''
    g = gradient(x)
    g_mag = np.sqrt(g[0]**2 + g[1]**2)
    m = g_mag >= mu
    m2 = (m == False) #bool(1-m)
    g[0][m] /= g_mag[m]
    g[1][m] /= g_mag[m]
    g[0][m2] /= mu
    g[1][m2] /= mu
    return -div(g)


def proj_l2(g, Lambda=1.0):
    '''
    Proximal operator of the L2,1 norm :
        L2,1(u) = sum_i ||u_i||_2
    i.e pointwise projection onto the L2 unit ball

    g : gradient-like numpy array
    Lambda : magnitude of the unit ball
    '''
    res = np.copy(g)
    n = np.maximum(np.sqrt(np.sum(g**2, 0))/Lambda, 1.0)
    res[0] /= n
    res[1] /= n
    return res



def norm2sq(mat):
    return np.dot(mat.ravel(), mat.ravel())

def norm1(mat):
    return np.sum(np.abs(mat))

def mydot(mat1, mat2):
    return np.dot(mat1.ravel(), mat2.ravel())


def generate_coords(img,center=None):
    if center is None: center = img.shape[0]/2, img.shape[1]/2
    R, C = np.mgrid[0:img.shape[0],0:img.shape[1]]
    R -= center[0]; C -= center[1]
    return R, C

def phantom_mask(img, radius=None):
    if radius is None: radius = img.shape[0]//2-10
    R, C = generate_coords(img)
    M = R**2+C**2
    res = np.zeros_like(img)
    res[M<radius**2] = img[M<radius**2]
    return res

def entropy(img):
    '''
    Computes the entropy of an image (similar to Matlab function)
    '''
    h, _ = np.histogram(img, 256)
    h = h.astype('f')
    h /= 1.0*img.size
    h[h == 0] = 1.0
    return -np.sum(h*np.log(h))

def KL(img1, img2):
    '''
    Computes the Kullback-Leibler divergence between two images
    Mind that this function is not symmetric. The second argument should be the "reference" image.
    '''
    x, _ = np.histogram(img1, 256)
    y, _ = np.histogram(img2, 256)
    m = (y != 0) # integers
    x_n, y_n = x[m], y[m]
    m = (x_n != 0)
    x_n, y_n = 1.0 * x_n[m], 1.0 * y_n[m]
    Sx, Sy = x.sum()*1.0, y.sum()*1.0
    return (1.0/Sx) * np.sum(x_n * np.log(x_n/y_n * Sy/Sx))





def mse(img1,img2):
   '''
   Computes the Mean Square Error between two images
   '''
   if img1.dtype != np.dtype('float64'): img1 = img1.astype('float64')
   if img2.dtype != np.dtype('float64'): img2 = img2.astype('float64')
   (n,m) = img1.shape
   if (n,m) != img2.shape:
      print('The images do not have the same size')
      return -1
   return np.sum(np.sum((img1[:,:] - img2[:,:])**2))/(m*n)
