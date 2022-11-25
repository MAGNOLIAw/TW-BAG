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
import os
import pickle
# Internal libraries/scripts
import data_loading.hcp_dti_sample as MIScnn_sample

#-----------------------------------------------------#
#                    Data IO class                    #
#-----------------------------------------------------#
# Class to handle all input and output functionality
class HCP_DTI_Data_IO:
    # Class variables
    interface = None                    # Data I/O interface
    input_path = None                   # Path to input data directory
    output_path = None                  # Path to MIScnn prediction directory
    batch_path = None                   # Path to temporary batch storage directory
    indices_list = None                 # List of sample indices after data set initialization
    delete_batchDir = None              # Boolean for deletion of complete tmp batches directory
                                        # or just the batch data for the current seed
    # seed = random.randint(0,99999999)   # Random seed if running multiple MIScnn instances
    seed = 6666
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating an object of the Data IO class.
    This class provides functionality for handling all input and output processes
    of the imaging data, as well as the temporary backup of batches to the disk.

    The user is only required to create an instance of the Data IO class with the desired specifications
    and IO interface for the correct format. It is possible to create a custom IO interface for handling
    special data structures or formats.

    Args:
        interface (io_interface):   A data IO interface which inherits the abstract_io class with the following methods:
                                    initialize, load_image, load_segmentation, load_prediction, save_prediction
        input_path (string):        Path to the input data directory, in which all imaging data have to be accessible.
        output_path (string):       Path to the output data directory, in which computed predictions will berun_evaluation.pyrun_evaluation.py stored. This directory
                                    will be created.
        batch_path (string):        Path to the batch data directory. This directory will be created and used for temporary files.
        delete_batchDir (boolean):  Boolean if the whole temporary batch directory for prepared batches should be deleted after
                                    model utilization. If false only the batches with the associated seed will be deleted.
                                    This parameter is important when running multiple instances of MIScnn.
    """
    def __init__(self, interface, input_path, output_path="predictions",
                 batch_path="batches", delete_batchDir=True,
                 img_path=None, mask_path=None, gt_path=None):
        # Parse parameter
        self.interface = interface
        self.input_path = input_path
        self.output_path = output_path
        self.batch_path = batch_path
        self.delete_batchDir = delete_batchDir
        # Initialize Data I/O interface
        self.indices_list = interface.initialize(input_path, img_path)
        self.img_path = img_path
        self.mask_path = mask_path
        self.gt_path = gt_path

    #---------------------------------------------#
    #                Sample Loader                #
    #---------------------------------------------#
    # Load a sample from the data set
    def sample_loader(self, index, load_seg=True, load_pred=False, backup=False, load_gt=True, load_eigen=False):
        # If sample is a backup -> load it from pickle
        if backup : return self.load_sample_pickle(index)
        # Load the image with the I/O interface
        image, aff = self.interface.load_image(index, self.img_path)
        # Create a Sample object
        sample = MIScnn_sample.HCPdtiSample(index, image, self.interface.channels,
                                      self.interface.classes, aff)
        # IF needed read the provided segmentation for current sample
        if load_seg:
            segmentation = self.interface.load_segmentation(index, self.mask_path)
            sample.add_segmentation(segmentation)
        # IF needed read the provided prediction for current sample
        if load_pred:
            prediction = self.interface.load_prediction(index, self.output_path)
            sample.add_prediction(prediction)
        if load_gt:
            # gt = image.copy()
            gt, aff = self.interface.load_gt(index, self.gt_path)
            sample.add_gt(gt)
        if load_eigen:
            eigenvalue1, eigenvalue2, eigenvalue3 = self.interface.load_eigen(index)
            sample.add_eigen(eigenvalue1, eigenvalue2, eigenvalue3)
        # Add optional details to the sample object
        sample.add_details(self.interface.load_details(index))
        # Return sample object
        return sample

    #---------------------------------------------#
    #              Prediction Backup              #
    #---------------------------------------------#
    # Save a segmentation prediction
    def save_prediction(self, pred, index, aff=None):
        # Create the output directory if not existent
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        # Backup the prediction
        self.interface.save_prediction(pred, index, self.output_path, aff)

    #---------------------------------------------#
    #                Sample Backup                #
    #---------------------------------------------#
    # Backup samples for later access
    def backup_sample(self, sample):
        if not os.path.exists(self.batch_path) : os.mkdir(self.batch_path)
        sample_path =cd
        os.path.join(self.batch_path, str(self.seed) + "." + \
                                   sample.index + ".pickle")
        if not os.path.exists(sample_path):
            with open(sample_path, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Back up sample in:', sample_path)

    # Load a backup sample from pickle
    def load_sample_pickle(self, index):
        sample_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                                   index + ".pickle")
        with open(sample_path,'rb') as reader:
            sample = pickle.load(reader)
        return sample

    #---------------------------------------------#
    #               Variable Access               #
    #---------------------------------------------#
    def get_indiceslist(self):
        return self.indices_list.copy()
