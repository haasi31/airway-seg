experiment_name=dataset_2023_11_15

echo "Copy labels to /home/ahaas/data/2_manual_adapted_data"
in_dir=/home/ahaas/data/1_simulated_data/$experiment_name
out_dir=/home/ahaas/data/2_manual_adapted_data/$experiment_name/labels
mkdir -p $out_dir
for mask_path in $in_dir/*; do
    cp $mask_path/airways/airways_label_minradius0.0011.nii.gz $out_dir/syn_$(basename $mask_path)_airways.nii.gz
done
