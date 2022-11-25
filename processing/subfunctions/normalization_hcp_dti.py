#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
# Internal libraries/scripts
from processing.subfunctions.abstract_subfunction import Abstract_Subfunction
import nibabel as nib
import os
#-----------------------------------------------------#
#          Subfunction class: Normalization           #
#-----------------------------------------------------#
""" A Normalization Subfunction class which normalizes the intensity pixel values of an image using
    the Z-Score technique (default setting), through scaling to [0,1] or to grayscale [0,255].

Args:
    mode (string):          Mode which normalization approach should be performed.
                            Possible modi: "z-score", "minmax" or "grayscale"

Methods:
    __init__                Object creation function
    preprocessing:          Pixel intensity value normalization the imaging data
    postprocessing:         Do nothing
"""
class NormalizationHCPdti(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score"):
        self.mode = mode
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        self.brain_img = None
        self.brain_mask = None
        self.ref_img = None
        self.ref_img_2 = None
        self.ref_mask = None
        self.lesion_mask = None

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True, validation=True, opt=None):

        # Access image, ground truth and prediction data
        brain_img = sample.img_data.squeeze().copy()
        brain_mask = sample.seg_data.squeeze()

        # ref mask: gt, healthy brain mask
        ref_img = sample.gt_data.squeeze().copy()
        ref_img_2 = sample.gt_data.squeeze().copy()
        ref_mask = ref_img.copy()[:,:,:,0]
        ref_mask[ref_mask != 0] = 1

        input_brain_mask = brain_img.copy()[:,:,:,0]
        input_brain_mask[input_brain_mask != 0] = 1

        crop_region_mask = np.zeros_like(input_brain_mask)
        crop_region_mask[input_brain_mask == 0] = 1
        crop_region_mask[brain_mask == 0] = 0

        # Perform z-score normalization
        if self.mode == "z-score":
            # if the slice does not have input brain region
            if brain_img[input_brain_mask == 1].shape[0] != 0:
                mean_along_channel = np.mean(brain_img[input_brain_mask == 1], axis=(0))
                std_along_channel = np.std(brain_img[input_brain_mask == 1], axis=(0))
            else:
                mean_along_channel = np.mean(ref_img[ref_mask == 1], axis=(0))
                std_along_channel = np.std(ref_img[ref_mask == 1], axis=(0))

            for c in range(brain_img.shape[3]):
                brain_img[:,:,:,c][input_brain_mask == 1] = (brain_img[:,:,:,c][input_brain_mask == 1] - mean_along_channel[c]) / std_along_channel[c]
                ref_img[:, :, :, c][ref_mask == 1] = (ref_img[:, :, :, c][ref_mask == 1] - mean_along_channel[c]) / std_along_channel[c]

                brain_img[:,:,:,c][input_brain_mask == 0] = 0
                ref_img[:, :, :, c][ref_mask == 0] = 0

            ref_mean = np.mean(ref_img[ref_mask == 1], axis=(0))
            ref_std = np.std(ref_img[ref_mask == 1], axis=(0))

            self.brain_mean = mean_along_channel
            self.brain_std = std_along_channel
            self.ref_mean = ref_mean
            self.ref_std = ref_std

        # Perform MinMax normalization between [0,1]
        elif self.mode == "minmax":
            # Identify minimum and maximum
            min_val = np.min(brain_img[brain_mask == 1])
            max_val = np.max(brain_img[brain_mask == 1])
            val_range = max_val - min_val
            self.min = min_val
            self.max = max_val
            # Scaling
            brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - min_val) / val_range
            brain_img[brain_mask == 0] = 0
            image_normalized = ref_img

        elif self.mode == "grayscale":
            # Identify minimum and maximum
            max_value = np.max(brain_img[brain_mask == 1])
            min_value = np.min(brain_img[brain_mask == 1])
            # Scaling
            ref_img[ref_mask != 0] = (ref_img[ref_mask == 1] - min_value) / (max_value - min_value)
            ref_img[ref_mask == 0] = 0
            image_scaled = ref_img
            image_normalized = np.around(image_scaled * 255, decimals=0)

        else : raise NameError("Subfunction - Normalization: Unknown modus")
        # Update the sample with the normalized image
        sample.img_data = brain_img
        sample.gt_data = ref_img
        self.brain_img = brain_img
        self.brain_mask = brain_mask
        self.ref_img = ref_img
        self.ref_img_2 = ref_img_2
        self.ref_mask = ref_mask
        self.sample = sample
        self.input_brain_mask = input_brain_mask
        self.crop_region_mask = crop_region_mask
    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
