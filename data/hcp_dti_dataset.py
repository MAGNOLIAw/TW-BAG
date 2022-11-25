import os
import torch
from data.base_dataset import BaseDataset
import numpy as np
import nibabel as nib

from utils.patch_operations import find_bounding_box, slice_matrix, concat_matrices, pad_patch, crop_patch, find_bounding_box
from processing.subfunctions import NormalizationHCPdti

class HCPdtiDataset(BaseDataset):
    """A dataset class for brain image dataset.

    It assumes that the directory '/path/to/data/train' contains brain image slices
    in torch.tensor format to speed up the I/O. Otherwise, you can load MRI brain images
    using nibabel package and preprocess the slices during loading period.
    """

    def __init__(self, opt):
        """
        Initialize this dataset class.
        """
        BaseDataset.__init__(self, opt)
        # print('HCP_DTI_Dataset is created.')
        self.sub_list = os.listdir(opt.dataroot)
        self.opt = opt

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

        self.patch_shape = (64, 64, self.opt.patch_axial)
        self.patchwise_overlap = (32, 32, self.opt.patch_axial//2)
        self.patchwise_skip_blanks = self.opt.patchwise_skip_blanks
        self.prepare_batches = False
        self.analysis = 'patchwise-grid'

        self.save_axial_slices = 80
        self.invalid_axial_slices = 30

        # set up training/validation
        shuffle_batches = opt.shuffle_batches
        if self.opt.phase == 'train':
            training = True
            validation = False
            self.sample_list_tr = opt.sample_list
            self.set_up(self.sample_list_tr, training, validation, shuffle_batches)
        elif self.opt.phase == 'test':
            training = False
            validation = False
            self.sample_list_ts = opt.sample_list
            self.set_up([self.sample_list_ts[opt.eval_sample]], training, validation, shuffle_batches)
            self.save_axial_slices = 0
            self.invalid_axial_slices = 0
        elif self.opt.phase == 'validation':
            training = False
            validation = True
            self.sample_list_ts = opt.sample_list
            self.set_up([self.sample_list_ts[opt.eval_sample]], training, validation, shuffle_batches)
            self.save_axial_slices = 0
            self.invalid_axial_slices = 0
        else:
            print('Invalid phase!')

    def set_up(self, sample_list, training=False, validation=False, shuffle=False):
        # Parse sample list
        if isinstance(sample_list, list) : self.sample_list = sample_list.copy()
        elif type(sample_list).__module__ == np.__name__ :
            self.sample_list = sample_list.tolist()
        else : raise ValueError("Sample list have to be a list or numpy array!")

        # Create a working environment from the handed over variables
        self.training = training
        self.validation = validation
        self.shuffle = shuffle

        self.coord_queue = []
        self.samples = {}
        self.cache = {}

        # for normalization
        gt_mean = []
        gt_std = []

        # print('---------------Load data and preprocessing-----------------')
        for i, index in enumerate(sample_list):
            # Load sample
            print('[loading case]:', index)
            if not self.prepare_batches:
                if not training and not validation:
                    # test: load without gt
                    sample = self.opt.data_io.sample_loader(index, load_seg=True, load_pred=False, backup=False,
                                                            load_gt=False)
                else:
                    sample = self.opt.data_io.sample_loader(index, load_seg=True, load_pred=False, backup=False,
                                                            load_gt=True)
            # Load sample from backup
            else:
                sample = self.opt.data_io.sample_loader(index, backup=True)

            # check if not all background
            if np.sum(sample.seg_data[:, :,
                      sample.seg_data.shape[2] - int(self.opt.save_axial * sample.seg_data.shape[2]):, :]) == 0:
                continue

            # crop top slices
            if not training:
                self.opt.save_axial = 1
                self.cache["shape_" + str(index)] = sample.img_data.shape
            sample.img_data = sample.img_data[:, :,
                              sample.img_data.shape[2] - int(self.opt.save_axial * sample.img_data.shape[2]):, :]
            sample.seg_data = sample.seg_data[:, :,
                              sample.seg_data.shape[2] - int(self.opt.save_axial * sample.seg_data.shape[2]):, :]
            sample.gt_data = sample.gt_data[:, :,
                             sample.gt_data.shape[2] - int(self.opt.save_axial * sample.gt_data.shape[2]):, :]

            # patch slicing before norm: run patchwise grid slicing, return coordinates of patches
            print('[starting patch slicing]...')
            coords_img_data = self.analysis_patchwise_grid(sample, training, index)
            self.coord_queue.extend(coords_img_data)
            print('crop patches from: ', i, index, sample.img_data.shape, len(coords_img_data),
                  len(self.coord_queue))

            # normalization
            print('[starting normalization]...')
            sf_zscore = NormalizationHCPdti(mode="z-score")
            # Assemble Subfunction classes into a list
            subfunctions = [sf_zscore]
            for sf in subfunctions:
                sf.preprocessing(sample, training=training, validation=validation, opt=self.opt)
                if str(sf.mode) == 'z-score' or 'z-score_v2' or 'z-score_v3':
                    gt_mean.append(sf.ref_mean)
                    gt_std.append(sf.ref_std)
                sample.input_brain_mask = sf.input_brain_mask
                sample.crop_region_mask = sf.crop_region_mask
            self.sf_zscore = sf_zscore
            self.subfunctions = subfunctions

            # save the mean and std of all cases during training
            if training:
                np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, sf.mode + '_gt_mean.npy'),
                        np.mean(np.array(gt_mean), axis=0), allow_pickle=True)
                np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, sf.mode + '_gt_std.npy'),
                        np.mean(np.array(gt_std), axis=0), allow_pickle=True)
            self.samples[index] = sample
        print('-----------end preprocessing-----------------')

    # ---------------------------------------------#
    #           Patch-wise grid Analysis          #
    # ---------------------------------------------#
    def analysis_patchwise_grid(self, sample, training, index=None):
        # Slice image into patches
        img_to_sliced = sample.img_data
        patches_img, coords_img = slice_matrix(img_to_sliced, self.patch_shape, self.patchwise_overlap,
                                               self.opt.data_io.interface.three_dim, index, save_coords=True)
        # Skip blank patches (only background)
        if training and self.patchwise_skip_blanks:
            # Iterate over each patch
            for i in reversed(range(0, len(patches_img))):
                # IF patch DON'T contain any non background class -> remove it
                if np.sum(patches_img[i]) == 0:
                    del patches_img[i]
                    del coords_img[i]
        # Concatenate a list of patches coordinates into a single numpy array
        coords_img_data = np.stack(coords_img, axis=0)
        # Return preprocessed data tuple
        return coords_img_data

    # ---------------------------------------------#
    #          Prediction Postprocessing          #
    # ---------------------------------------------#
    # Postprocess prediction data
    def postprocessing(self, sample, prediction, shape=None, coords=None):
        # Reassemble patches into original shape for patchwise analysis
        if self.analysis == "patchwise-crop" or \
                self.analysis == "patchwise-grid":
            # Check if patch was padded
            slice_key = "slicer_" + str(sample)
            if slice_key in self.cache:
                prediction = crop_patch(prediction, self.cache[slice_key])
            # Load cached shape & Concatenate patches into original shape
            seg_shape = self.cache.pop("shape_" + str(sample))
            prediction = concat_matrices(patches=prediction,
                                         image_size=seg_shape,
                                         window=self.patch_shape,
                                         overlap=self.patchwise_overlap,
                                         three_dim=self.opt.data_io.interface.three_dim,
                                         coords=coords)
        # For fullimages remove the batch axis
        else:
            prediction = np.squeeze(prediction, axis=0)
        # Run Subfunction postprocessing on the prediction
        for sf in reversed(self.subfunctions):
            prediction = sf.postprocessing(prediction)
        # Return postprocessed prediction
        return prediction

    # Return the next batch for associated index
    def __getitem__(self, idx):

        self.coords_batches = self.coord_queue

        self.now_sample = self.samples[self.coords_batches[idx]['index']]

        x_start = self.coords_batches[idx]['x_start']
        x_end = self.coords_batches[idx]['x_end']
        y_start = self.coords_batches[idx]['y_start']
        y_end = self.coords_batches[idx]['y_end']
        z_start = self.coords_batches[idx]['z_start']
        z_end = self.coords_batches[idx]['z_end']


        x = torch.from_numpy(self.now_sample.img_data[x_start:x_end, y_start:y_end, z_start:z_end])
        y = torch.from_numpy(self.now_sample.seg_data[x_start:x_end, y_start:y_end, z_start:z_end])
        gt = torch.from_numpy(self.now_sample.gt_data[x_start:x_end, y_start:y_end, z_start:z_end])

        # w h d c
        brain = x.clone()
        mask = y.clone()
        # w h d c -> c w h d
        brain = brain.permute(3,0,1,2)
        mask = mask.permute(3, 0, 1, 2)
        gt = gt.permute(3,0,1,2)

        input_brain_mask = torch.from_numpy(self.now_sample.input_brain_mask[x_start:x_end, y_start:y_end, z_start:z_end])
        crop_region_mask = torch.from_numpy(self.now_sample.crop_region_mask[x_start:x_end, y_start:y_end, z_start:z_end])

        input_brain_mask = torch.unsqueeze(input_brain_mask, 0)
        input_brain_mask = torch.repeat_interleave(input_brain_mask, 6, dim=0)
        crop_region_mask = torch.unsqueeze(crop_region_mask, 0)
        crop_region_mask = torch.repeat_interleave(crop_region_mask, 6, dim=0)

        ret = {}
        ret['brain'] = brain
        ret['mask'] = mask
        ret['gt'] = gt
        ret['crop_region_mask'] = crop_region_mask
        return ret

    # Return the number of batches for one epoch
    def __len__(self):
        return len(self.coord_queue)

    def modify_commandline_options(parser, is_train):
        """
        Add any new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--test_fold', default='1', type=int, help='test fold number')
        parser.add_argument('--shuffle_batches', default=True, type=bool, help='whether batch order should be shuffled or not ?')
        parser.add_argument('--save_axial', default=1, type=float, help='proportion of saved axial slices')
        parser.add_argument('--patch_axial', default=32, type=int, help='patch shape of axial dimension')

        parser.add_argument('--img_path', type=str, default=None, help='path to input')
        parser.add_argument('--mask_path', type=str, default=None, help='path to mask')
        parser.add_argument('--gt_path', type=str, default=None, help='path to gt')
        return parser
