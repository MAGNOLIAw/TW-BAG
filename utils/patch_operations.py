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
#External libraries
import numpy as np
import math
from batchgenerators.augmentations.utils import pad_nd_image

#-----------------------------------------------------#
#      Pad and crop patch to desired patch shape      #
#-----------------------------------------------------#
def pad_patch(patch, patch_shape, return_slicer=False):
    # Initialize stat length to overwrite batchgenerators default
    kwargs = {"stat_length": None}
    # Transform prediction from channel-last to channel-first structure
    patch = np.moveaxis(patch, -1, 1)
    # Run padding
    padding_results = pad_nd_image(patch, new_shape=patch_shape,
                                   mode="minimum", return_slicer=return_slicer,
                                   kwargs=kwargs)
    # Return padding results
    if return_slicer:
        # Transform data from channel-first back to channel-last structure
        padded_patch = np.moveaxis(padding_results[0], 1, -1)
        return padded_patch, padding_results[1]
    else:
        # Transform data from channel-first back to channel-last structure
        padding_results = np.moveaxis(padding_results, 1, -1)
        return padding_results

def crop_patch(patch, slicer):
    # Transform prediction from channel-last to channel-first structure
    patch = np.moveaxis(patch, -1, 1)
    # Exclude the number of batches and classes from the slice range
    slicer[0] = slice(None)
    slicer[1] = slice(None)
    # Crop patches according to slicer
    patch_cropped = patch[tuple(slicer)]
    # Transform data from channel-first back to channel-last structure
    patch_cropped = np.moveaxis(patch_cropped, 1, -1)
    # Return cropped patch
    return patch_cropped

#-----------------------------------------------------#
#         Slice and Concatenate Function Hubs         #
#-----------------------------------------------------#
# Slice a matrix
def slice_matrix(array, window, overlap, three_dim, index=None, save_coords=False):
    if three_dim: return slice_3Dmatrix(array, window, overlap, index, save_coords)
    else: return slice_2Dmatrix(array, window, overlap)

# Concatenate a matrix
def concat_matrices(patches, image_size, window, overlap, three_dim, coords=None):
    if three_dim: return concat_3Dmatrices(patches, image_size, window, overlap, coords)
    else: return concat_2Dmatrices(patches, image_size, window, overlap)

#-----------------------------------------------------#
#          Slice and Concatenate 2D Matrices          #
#-----------------------------------------------------#
# Slice a 2D matrix
def slice_2Dmatrix(array, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) /
                            float(window[1] - overlap[1])))

    # Iterate over it x,y
    patches = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):

            # Define window edges
            x_start = x*window[0] - x*overlap[0]
            x_end = x_start + window[0]
            y_start = y*window[1] - y*overlap[1]
            y_end = y_start + window[1]
            # Adjust ends
            if(x_end > len(array)):
                # Create an overlapping patch for the last images / edges
                # to ensure the fixed patch/window sizes
                x_start = len(array) - window[0]
                x_end = len(array)
                # Fix for MRIs which are smaller than patch size
                if x_start < 0 : x_start = 0
            if(y_end > len(array[0])):
                y_start = len(array[0]) - window[1]
                y_end = len(array[0])
                # Fix for MRIs which are smaller than patch size
                if y_start < 0 : y_start = 0
            # Cut window
            window_cut = array[x_start:x_end,y_start:y_end]
            # Add to result list
            patches.append(window_cut)
    return patches

# Concatenate a list of patches together to a numpy matrix
def concat_2Dmatrices(patches, image_size, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                            float(window[1] - overlap[1])))

    # Iterate over it x,y,z
    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            # Calculate pointer from 2D steps to 1D list of patches
            pointer = x*steps_y + y
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = patches[pointer]
            else:
                matrix_p = patches[pointer]
                # Handle y-axis overlap
                slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                  image_size, window, 1)
                matrix_y, matrix_p = handle_overlap(matrix_y, matrix_p,
                                                    slice_overlap,
                                                    axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_p), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Handle x-axis overlap
            slice_overlap = calculate_overlap(x, steps_x, overlap,
                                              image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                slice_overlap,
                                                axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return(matrix_x)

#-----------------------------------------------------#
#          Slice and Concatenate 3D Matrices          #
#-----------------------------------------------------#
# Slice a 3D matrix
def slice_3Dmatrix(array, window, overlap, index=None, save_coords=False):
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) /
                            float(window[1] - overlap[1])))
    steps_z = int(math.ceil((len(array[0][0]) - overlap[2]) /
                            float(window[2] - overlap[2])))

    # Iterate over it x,y,z
    patches = []
    coords = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Define window edges
                x_start = x*window[0] - x*overlap[0]
                x_end = x_start + window[0]
                y_start = y*window[1] - y*overlap[1]
                y_end = y_start + window[1]
                z_start = z*window[2] - z*overlap[2]
                z_end = z_start + window[2]
                # Adjust ends
                if(x_end > len(array)):
                    # Create an overlapping patch for the last images / edges
                    # to ensure the fixed patch/window sizes
                    x_start = len(array) - window[0]
                    x_end = len(array)
                    # Fix for MRIs which are smaller than patch size
                    if x_start < 0 : x_start = 0
                if(y_end > len(array[0])):
                    y_start = len(array[0]) - window[1]
                    y_end = len(array[0])
                    # Fix for MRIs which are smaller than patch size
                    if y_start < 0 : y_start = 0
                if(z_end > len(array[0][0])):
                    z_start = len(array[0][0]) - window[2]
                    z_end = len(array[0][0])
                    # Fix for MRIs which are smaller than patch size
                    if z_start < 0 : z_start = 0
                # Cut window
                window_cut = array[x_start:x_end,y_start:y_end,z_start:z_end]
                # Add to result list
                patches.append(window_cut)

                coord_dict = {
                    'index': index,
                    'x_start': x_start,
                    'x_end': x_end,
                    'y_start': y_start,
                    'y_end': y_end,
                    'z_start': z_start,
                    'z_end': z_end
                }
                coords.append(coord_dict)

    if not save_coords:
        return patches
    else:
        return patches, coords

# Concatenate a list of patches together to a numpy matrix
def concat_3Dmatrices(patches, image_size, window, overlap, coords=None):
    if coords is None:
        # Calculate steps
        steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                                float(window[0] - overlap[0])))
        steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                                float(window[1] - overlap[1])))
        steps_z = int(math.ceil((image_size[2] - overlap[2]) /
                                float(window[2] - overlap[2])))

        # Iterate over it x,y,z
        matrix_x = None
        matrix_y = None
        matrix_z = None
        pointer = 0

        counts = np.zeros((steps_x, steps_y, steps_z))
        for x in range(0, steps_x):
            for y in range(0, steps_y):
                for z in range(0, steps_z):
                    # Calculate pointer from 3D steps to 1D list of patches
                    pointer = z + y*steps_z + x*steps_y*steps_z
                    # Connect current patch to temporary Matrix Z
                    if z == 0:
                        matrix_z = patches[pointer]
                    else:
                        matrix_p = patches[pointer]
                        # Handle z-axis overlap
                        counts[x,y,z]+=1
                        slice_overlap = calculate_overlap(z, steps_z, overlap,
                                                          image_size, window, 2)
                        matrix_z, matrix_p = handle_overlap(matrix_z, matrix_p,
                                                            slice_overlap,
                                                            axis=2)
                        matrix_z = np.concatenate((matrix_z, matrix_p),
                                                  axis=2)
                # Connect current tmp Matrix Z to tmp Matrix Y
                if y == 0:
                    matrix_y = matrix_z
                else:
                    # Handle y-axis overlap
                    counts[x, y, z] += 1
                    slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                      image_size, window, 1)
                    matrix_y, matrix_z = handle_overlap(matrix_y, matrix_z,
                                                        slice_overlap,
                                                        axis=1)
                    matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)
            # Connect current tmp Matrix Y to final Matrix X
            if x == 0:
                matrix_x = matrix_y
            else:
                # Handle x-axis overlap
                counts[x, y, z] += 1
                slice_overlap = calculate_overlap(x, steps_x, overlap,
                                                  image_size, window, 0)
                matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                    slice_overlap,
                                                    axis=0)
                matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
        # Return final combined matrix
        return (matrix_x)
    else:
        img = np.zeros(image_size, dtype=np.float32)
        counts = np.zeros(image_size, dtype=np.float32)
        for ci, coord in enumerate(coords):
            sub_index = coord['sub']
            x_start = coord['x_start']
            x_end = coord['x_end']
            y_start = coord['z_start']
            y_end = coord['z_end']
            z_start = coord['y_start']
            z_end = coord['y_end']
            dataloader_idx = coord['dataloader_idx']
            if dataloader_idx != ci:
                print('patches were shuffle, please be careful!')
            patch = patches[dataloader_idx, :, :, :]
            img[x_start:x_end, y_start:y_end, z_start:z_end] += patch
            counts[x_start:x_end, y_start:y_end, z_start:z_end] += 1

        counts[counts == 0] = 1
        img /= counts
        return img

#-----------------------------------------------------#
#          Subroutines for the Concatenation          #
#-----------------------------------------------------#
# Calculate the overlap of the current matrix slice
def calculate_overlap(pointer, steps, overlap, image_size, window, axis):
            # Overlap: IF last axis-layer -> use special overlap size
            if pointer == steps-1 and not (image_size[axis]-overlap[axis]) \
                                            % (window[axis]-overlap[axis]) == 0:
                current_overlap = window[axis] - \
                                  (image_size[axis] - overlap[axis]) % \
                                  (window[axis] - overlap[axis])
            # Overlap: ELSE -> use default overlap size
            else:
                current_overlap = overlap[axis]
            # Return overlap
            return current_overlap

# Handle the overlap of two overlapping matrices
def handle_overlap(matrixA, matrixB, overlap, axis):
    # Access overllaping slice from matrix A
    idxA = [slice(None)] * matrixA.ndim
    matrixA_shape = matrixA.shape
    idxA[axis] = range(matrixA_shape[axis] - overlap, matrixA_shape[axis])
    sliceA = matrixA[tuple(idxA)]
    # Access overllaping slice from matrix B
    idxB = [slice(None)] * matrixB.ndim
    idxB[axis] = range(0, overlap)
    sliceB = matrixB[tuple(idxB)]
    # Calculate Average prediction values between the two matrices
    # and save them in matrix A
    matrixA[tuple(idxA)] = np.mean(np.array([sliceA, sliceB]), axis=0)
    # Remove overlap from matrix B
    matrixB = np.delete(matrixB, [range(0, overlap)], axis=axis)
    # Return processed matrices
    return matrixA, matrixB

def find_bounding_box(brain):
    x, y, z = brain.shape[0], brain.shape[1], brain.shape[2]
    for i in range(z):
        slice = brain[:,:,i]
        if np.sum(slice) > 0:
            save_z_from_I = i
            break

    for i in reversed(range(z)):
        slice = brain[:,:,i]
        if np.sum(slice) > 0:
            save_z_from_S = i
            break

    for i in range(y):
        slice = brain[:, i, :]
        if np.sum(slice) > 0:
            save_y_from_P = i
            break

    for i in reversed(range(y)):
        slice = brain[:, i, :]
        if np.sum(slice) > 0:
            save_y_from_A = i
            break

    for i in range(x):
        slice = brain[i,:,:]
        if np.sum(slice) > 0:
            save_x_from_L = i
            break

    for i in reversed(range(x)):
        slice = brain[i,:,:]
        if np.sum(slice) > 0:
            save_x_from_R = i
            break

    return save_x_from_L, save_x_from_R, save_y_from_P, save_y_from_A, save_z_from_I, save_z_from_S

