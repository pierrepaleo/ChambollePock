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
from image_operators import *
from utils import *
from tomo_operators import AstraToolbox

# -----
VERBOSE = True
# -----


def conjugate_gradient_TV(P, PT, sino, Lambda, mu=1e-4, n_it=300):
    '''
    Conjugate Gradient algorithm to minimize the objective function
        0.5*||P*x - d||_2^2 + Lambda*TV_mu (x)

    P : projection operator
    PT : backprojection operator
    sino: acquired data as a sinogram
    Lambda : parameter weighting the TV regularization
    mu : parameter of Moreau-Yosida approximation of TV (small positive value)
    n_it : number of iterations
    '''

    x = 0*PT(sino) # start from 0
    grad_f = -PT(sino)
    grad_F = grad_f + Lambda*grad_tv_smoothed(x, mu)
    d = -np.copy(grad_F)
    en = np.zeros(n_it)
    for k in range(0, n_it):
        grad_f_old = grad_f
        grad_F_old = grad_F
        ATAd = PT(P(d))
        # Calculate step size
        alpha = mydot(d, -grad_F_old)/mydot(d, ATAd)
        # Update variables
        x = x + alpha*d
        grad_f = grad_f_old + alpha*ATAd
        grad_F = grad_f + Lambda*grad_tv_smoothed(x,mu)
        beta = mydot(grad_F, grad_F - grad_F_old)/norm2sq(grad_F_old) # Polak-Ribiere
        if beta < 0:
            beta = 0
        d = -grad_F + beta*d
        # Energy
        fid = norm2sq(P(x)-sino)
        tv = tv_smoothed(x,mu)
        eng = fid+Lambda*tv
        en[k] = eng
        if VERBOSE and (k % 10 == 0):
            print("%d : Energy = %e \t Fid = %e\t TV = %e" %(k, eng, fid, tv))
        # Stoping criterion
        if np.abs(alpha) < 1e-34: # TODO : try other bounds
            print("Warning : minimum step reached, interrupting at iteration %d" %k)
            break;
    return en, x





if __name__ == "__main__":

    # Parameters
    # ----------
    n_angles = 80 # number of proj. angles
    Lambda = 0.64 # weight of TV regularization
    mu = 1e-10 # parameter of TV smoothing
    n_it = 300 # number of iterations

    # Init.
    # ------

    from scipy.misc import lena
    l = lena().astype('f')
    l = phantom_mask(l)
    AST = AstraToolbox(l.shape[0], n_angles)
    P = lambda x : AST.proj(x) *3.14159/2.0/n_angles # so that PT(P(x)) ~= x
    PT = lambda y : AST.backproj(y, filt=True)

    # Run
    # -----
    sino = P(l)
    en, rec = conjugate_gradient_TV(P, PT, sino, Lambda, mu, n_it)

    my_imshow((PT(sino), rec), shape=(1,2), cmap="gray")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(en)
    plt.show()



    AST.cleanup()

