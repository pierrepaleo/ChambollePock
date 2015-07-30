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
import matplotlib.pyplot as plt

def my_imshow(img_list, shape=None, cmap=None, nocbar=False):
    if isinstance(img_list, np.ndarray):
        is_array = True
        plt.figure()
        plt.imshow(img_list, interpolation="nearest", cmap=cmap)
        if nocbar is False: plt.colorbar()
        plt.show()

    elif shape:
        num = np.prod(shape)
        #~ if num > 1 and is_array:
            #~ print('Warning (my_imshow): requestred to show %d images but only one image was provided' %(num))
        if num != len(img_list):
            raise Exception('ERROR (my_imshow): requestred to show %d images but %d images were actually provided' %(num , len(img_list)))

        plt.figure()
        for i in range(0, num):
            curr = str(shape + (i+1,))
            curr = curr[1:-1].replace(',','').replace(' ','')
            if i == 0: ax0 = plt.subplot(curr)
            else: plt.subplot(curr, sharex=ax0, sharey=ax0)
            plt.imshow(img_list[i], interpolation="nearest", cmap=cmap)
            if nocbar is False: plt.colorbar()
        plt.show()


