experiment_name=dataset_2023_11_15
num_samples=20
scans="010 027 221"

echo "Copy masks to /home/ahaas/data/2_manual_adapted_data"
in_dir=/home/ahaas/data/1_simulated_data/$experiment_name
out_dir=/home/ahaas/data/2_manual_adapted_data/$experiment_name/masks
mkdir -p $out_dir
for mask_path in $in_dir/*; do
    cp $mask_path/airways/lung.nii.gz $out_dir/syn_$(basename $mask_path)_mask.nii.gz
done