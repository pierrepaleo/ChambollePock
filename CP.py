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
from math import sqrt
from image_operators import *
from utils import *
from tomo_operators import AstraToolbox
from convo_operators import *

# ----
VERBOSE = 1
# ----

def power_method(P, PT, data, n_it=10):
    '''
    Calculates the norm of operator K = [grad, P],
    i.e the sqrt of the largest eigenvalue of K^T*K = -div(grad) + P^T*P :
        ||K|| = sqrt(lambda_max(K^T*K))

    P : forward projection
    PT : back projection
    data : acquired sinogram
    '''
    x = PT(data)
    for k in range(0, n_it):
        x = PT(P(x)) - div(gradient(x))
        s = sqrt(norm2sq(x))
        x /= s
    return sqrt(s)



def chambolle_pock(P, PT, data, Lambda, L,  n_it, return_energy=True):
    '''
    Chambolle-Pock algorithm for the minimization of the objective function
        ||P*x - d||_2^2 + Lambda*TV(x)

    P : projection operator
    PT : backprojection operator
    Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
    L : norm of the operator [P, Lambda*grad] (see power_method)
    n_it : number of iterations
    return_energy: if True, an array containing the values of the objective function will be returned
    '''

    sigma = 1.0/L
    tau = 1.0/L

    x = 0*PT(data)
    p = 0*gradient(x)
    q = 0*data
    x_tilde = 0*x
    theta = 1.0

    if return_energy: en = np.zeros(n_it)
    for k in range(0, n_it):
        # Update dual variables
        p = proj_l2(p + sigma*gradient(x_tilde), Lambda)
        q = (q + sigma*P(x_tilde) - sigma*data)/(1.0 + sigma)
        # Update primal variables
        x_old = x
        x = x + tau*div(p) - tau*PT(q)
        x_tilde = x + theta*(x - x_old)
        # Calculate norms
        if return_energy:
            fidelity = 0.5*norm2sq(P(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (VERBOSE and k%10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_energy: return en, x
    else: return x





def chambolle_pock_relaxed(P, PT, data, Lambda, L,  n_it, return_energy=True):
    '''
    Theoritically faster than Chambolle-Pock, but uses more memory
    '''

    tau = 0.1
    sigma = 1.0/(tau*L**2)
    rho = 1.9

    x = 0*PT(data)
    x_tilde = 0*x
    u_p = 0*gradient(x)
    u_q = 0*P(x)
    u_tilde_p = 0*u_p
    u_tilde_q = 0*u_q

    if return_energy: en = np.zeros(n_it)
    for k in range(0, n_it):
        # x_tilde = prox_{tau*f}(x - tau*Kstar(u))
        x_tilde = x + tau*div(u_p) - tau*PT(u_q)
        # u_tilde = prox_{sigma*gstar}(u + sigma*K(2*x_tilde - x))
        u_tilde_p = proj_linf(u_p + sigma*gradient(2*x_tilde - x), Lambda)
        u_tilde_q = (u_q + sigma*P(2*x_tilde - x) - sigma*data)/(1.0 + sigma)
        # x = x + rho*(x_tilde - x)
        x = x + rho*(x_tilde - x)
        # u = u + rho*(u_tilde - u)
        u_p = u_p + rho*(u_tilde_p - u_p)
        u_q = u_q + rho*(u_tilde_q - u_q)
        # Calculate norms
        if return_energy:
            fidelity = 0.5*1.0*norm2sq(P(x)-data)
            tv = norm1(gradient(x))
            energy = 1.0*fidelity + Lambda*tv
            en[k] = energy
            if (VERBOSE and k%10 == 0):
                print("[%d] : energy %e \t fidelity %e \t TV %e" %(k,energy,fidelity,tv))
    if return_energy: return en, x
    else: return x



if __name__ == "__main__":

    # Parameters
    # ----------
    n_angles = 80 # number of proj. angles
    beta = 1.0 # weight of TV regularization
    n_it = 500 # number of iterations

    # Init.
    # ------
    from scipy.misc import lena
    l = lena().astype('f')
    l = phantom_mask(l)

    #~ from phantoms import phantom
    #~ l = phantom(512)[::-1,:]
    AST = AstraToolbox(l.shape[0], n_angles)
    P = lambda x : AST.proj(x) *3.14159/2.0/n_angles # so that PT(P(x)) ~= x
    PT = lambda y : AST.backproj(y, filt=True)

    # Uncomment to see deconvolution instead of tomographic reconstruction
    '''
    kern = gaussian1D(2.6)
    K = ConvolutionOperator(kern)
    P = lambda x : K*x
    PT = lambda x : K.T()*x
    beta = 0.05
    '''

    # Run
    # -----
    sino = P(l)
    L = power_method(P, PT, sino, n_it=100)# * 1.5
    print("||K|| = %f" % L)
    en, rec = chambolle_pock(P, PT, sino, beta, L, n_it)

    # Visualize
    # --------
    my_imshow((PT(sino), rec), shape=(1,2), cmap="gray", nocbar=True)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(en)
    plt.show()

    # Clear
    # ------

    AST.cleanup()



