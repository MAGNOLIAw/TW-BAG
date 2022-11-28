set -ex
python3 /home/xinyi/TW-BAG/test_hcp_dti.py \
--checkpoints_dir /home/xinyi/checkpoints3 \
--output_dir /home/xinyi/predictions_HCP_DTI \
--name HCP_DTI_twbag_zscore_0445 \
--model hcp_dti_mask \
--input_nc 7 \
--output_nc 6 \
--init_type kaiming \
--dataset_mode hcp_dti \
--batch_size 4 \
--gpu_ids 0 \
--conv_type TWBAG \
--phase validation \
--patch_axial 64 \
--img_path /home/data/HCP_DTI/HCP_tensor_cropped/test \
--mask_path /home/data/HCP_DTI/HCP_brain_mask/test \
--gt_path /home/data/HCP_DTI/HCP_tensor_gt/test \
