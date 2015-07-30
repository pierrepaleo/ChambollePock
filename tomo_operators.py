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
import astra

class AstraToolbox:

    def __init__(self, n_pixels, n_angles, rayperdetec=None):
        '''
        Initialize the ASTRA toolbox with a simple parallel configuration.
        The image is assumed to be square, and the detector count is equal to the number of rows/columns.
        '''
        self.vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, n_pixels, np.linspace(0,np.pi,n_angles,False))
        self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom)
        #~ self.rec_id = astra.data2d.create('-vol', self.vol_geom)

    def backproj(self, sino_data, filt=False):
        if filt is True:
            bid, rec = astra.create_backprojection(self.filter_projections(sino_data), self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        else:
            bid, rec = astra.create_backprojection(sino_data, self.proj_id)#,useCUDA=True) #last keyword for astra 1.1
        astra.data2d.delete(bid)
        return rec

    def proj(self, slice_data):
        sid, proj_data = astra.create_sino(slice_data, self.proj_id)
        astra.data2d.delete(sid)
        return proj_data

    def filter_projections(self, proj_set):
        nb_angles, l_x = proj_set.shape
        ramp = 1./l_x * np.hstack((np.arange(l_x), np.arange(l_x, 0, -1)))
        return np.fft.ifft(ramp * np.fft.fft(proj_set, 2*l_x, axis=1), axis=1)[:,:l_x].real

    def cleanup(self):
        #~ astra.data2d.delete(self.rec_id)
        astra.data2d.delete(self.proj_id)
