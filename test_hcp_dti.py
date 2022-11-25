"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.stats import *

from data_loading.interfaces import HCP_DTI_NIFTI_interface
from data_loading import HCP_DTI_Data_IO
from processing.subfunctions import NormalizationHCPdti

import torch
import subprocess
import time
from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn as nn
import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# device = torch.device('cpu')

torch.manual_seed(1)
np.random.seed(1)
random.seed(1)
torch.cuda.manual_seed_all(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    # opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

    # Initialize Data IO Interface for NIfTI data
    interface = HCP_DTI_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc)
    # Create Data IO object to load and write samples in the file structure
    data_io = HCP_DTI_Data_IO(interface, input_path=opt.dataroot, output_path=opt.output_dir+"/predictions_"+str(opt.name),
                                    batch_path="fod_norm_tensor", delete_batchDir=False,
                                    img_path=opt.img_path, mask_path=opt.mask_path, gt_path=opt.gt_path)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    sample_list.sort()

    # Create a summary file
    if not os.path.exists(data_io.output_path):
        subprocess.call(['mkdir', data_io.output_path])
    if os.path.exists(os.path.join(data_io.output_path, 'summary.txt')):
        subprocess.call(['rm', os.path.join(data_io.output_path, 'summary.txt')])
    summary_file = open(os.path.join(data_io.output_path, 'summary.txt'), 'w')

    entry = 'Load checkpoint from: ' + opt.checkpoints_dir + opt.name + '\n'

    torch.cuda.empty_cache()

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()

    # Create normalization
    sf_zscore = NormalizationHCPdti(mode="z-score")
    sf = [sf_zscore]

    opt.patchwise_skip_blanks = True
    opt.sample_list = [sample_list[0]]
    sample_list = opt.sample_list
    opt.eval_sample = 0
    opt.data_io = data_io
    save_prediction = True

    loss_list = []
    mse_lst = []
    mae_lst = []
    psnr = []
    ssim = []
    whole_ssim = []
    mse_med = []
    psnr_med = []
    ssim_med = []
    time_spent = []


    all_ref_mean = np.load('../all_ref_mean_mask.npy', allow_pickle=True)
    all_ref_std = np.load('../all_ref_mean_mask.npy', allow_pickle=True)
    tr_ref_mean = np.mean(all_ref_mean, axis=0)
    tr_ref_std = np.mean(all_ref_std, axis=0)
    for i in range(len(sample_list)):
        start_time = time.time()

        index = sample_list[i]
        entry += ('case: ' + index + '\n')

        # Create dataloader: do normalize + slice patches
        opt.eval_sample = i
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

        # Get normalized images
        sf_zscore = dataset.dataset.sf_zscore
        brain_img = sf_zscore.brain_img
        brain_mask = sf_zscore.brain_mask
        ref_img = sf_zscore.ref_img
        ref_img_2 = sf_zscore.ref_img_2
        ref_mask = sf_zscore.ref_mask
        mean = sf_zscore.mean
        std = sf_zscore.std
        sample = sf_zscore.sample

        tr_ref_mean = sf_zscore.brain_mean
        tr_ref_std = sf_zscore.brain_std

        pred_list = []
        val_loss = 0.
        for j, data in enumerate(dataset):
            model.set_input(data)
            output = model.test()  # run inference
            losses = model.get_current_losses()
            val_loss += losses['R']

            output = output.detach().cpu().numpy()
            # b c w h d -> b w h d c
            output = np.transpose(output, (0, 2, 3, 4, 1))
            temp = output
            pred_list.append(temp)
            # print('end inference')

        # Postprocess predicted patches
        pred_seg = np.concatenate(pred_list, axis=0)
        pred_seg = dataset.dataset.postprocessing(sample_list[opt.eval_sample], pred_seg)

        # Compute testing loss
        loss_list.append(val_loss / (j + 1))
        print('[Test] loss:', val_loss / (j + 1))

        # Compute result image
        result_img = pred_seg.squeeze()
        print(result_img.shape)

        brain_mask = interface.load_segmentation(index, opt.mask_path)

        # Normalization
        for c in range(result_img.shape[3]):
            mean = tr_ref_mean[c]
            std = tr_ref_std[c]
            result_img[:, :, :, c][brain_mask == 1] = result_img[:, :, :, c][brain_mask == 1] * std + mean

        # generate masks
        input, _ = interface.load_image(index, opt.img_path)
        input_brain_mask = input.copy()
        input_brain_mask[input_brain_mask != 0] = 1

        crop_region_mask = np.zeros_like(input_brain_mask)
        crop_region_mask[input_brain_mask == 0] = 1
        crop_region_mask[brain_mask == 0] = 0

        result_img = input * (1 - crop_region_mask) + result_img * crop_region_mask

        gt, _ = interface.load_gt(index, opt.gt_path)
        ref_patch = gt * crop_region_mask
        result_patch = result_img * crop_region_mask
        valid_no = np.sum(crop_region_mask)  # valid pixels

        # compute metrics
        print(ref_img.shape, result_img.shape)
        whole_ssim.append(((measure.compare_ssim(ref_img, result_img, multichannel=True))))
        ssim.append((1 - (measure.compare_ssim(ref_img, result_img, multichannel=True))) * 100)

        # compute metrics
        org_mae = np.sum(np.abs(ref_patch - result_patch)) / valid_no
        mae_lst.append(org_mae)
        sub_mae, sub_mse, sub_psnr = psnr2(ref_patch, result_patch, valid_no)
        entry += ('MSE: ' + str(sub_mse) + ' PSNR: ' + str(sub_psnr) + '\n')
        mse_lst.append(sub_mse)
        psnr.append(sub_psnr)

        time_spent.append(time.time() - start_time)
        print('MSE: ', sub_mse, 'PSNR: ', sub_psnr)

        # Backup predicted segmentation
        if save_prediction:
            dataset.dataset.opt.data_io.save_prediction(result_img, sample_list[i], aff=sample.aff)
        print('***************************End inference******************************')

    mean_mse_info = 'mean mse: {:.6f}±{:.6f}'.format(np.mean(mse_lst), np.std(mse_lst))
    mean_mae_info = 'mean mae: {:.6f}±{:.6f}'.format(np.mean(mae_lst), np.std(mae_lst))
    mean_psnr_info = 'mean psnr: {:.4f}±{:.4f}'.format(np.mean(psnr), np.std(psnr))
    mean_ssim_info = 'mean ssim diff: {:.4f}±{:.4f}'.format(np.mean(ssim), np.std(ssim))
    mean_time = 'average case time: {:.4f} seconds'.format(np.mean(time_spent))
    print(mean_mse_info)
    print(mean_mae_info)
    print(mean_psnr_info)
    print(mean_ssim_info)
    print(mean_time)

    print('For record keeping: \n {:.4f}±{:.4f} {:.4f}±{:.4f} {:.4f}±{:.4f} {:.4f}±{:.4f}' \
          .format(np.mean(mse_lst), np.std(mse_lst), np.mean(mae_lst), np.std(mae_lst), \
                  np.mean(psnr), np.std(psnr), np.mean(ssim), np.std(ssim)))

    entry += (mean_mse_info + '\n')
    entry += (mean_psnr_info + '\n')
    entry += (mean_ssim_info + '\n')
    entry += '\n'

    summary_file.write(entry)
    summary_file.close()
