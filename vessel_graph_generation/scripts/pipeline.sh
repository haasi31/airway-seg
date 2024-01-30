#!/bin/bash

experiment_name=dataset_2023_11_15
num_samples=20
scans="010 027 221"

# echo "Run airway graph simulation"
# python /home/ahaas/airway-seg/generate_vessel_graph.py \
#     --config_file /home/ahaas/airway-seg/vessel_graph_generation/configs/airways_config.yml \
#     --num_samples $num_samples \
#     --threads 14 \
#     --experiment_name $experiment_name \
#     --scans $scans

# echo "Run vessel graph simulation"
# python /home/ahaas/airway-seg/generate_vessel_graph.py \
#     --config_file /home/ahaas/airway-seg/vessel_graph_generation/configs/vessels_config.yml \
#     --num_samples $num_samples \
#     --threads 14 \
#     --experiment_name $experiment_name \
#     --scans $scans

# echo "Run voxelization"
# python /home/ahaas/airway-seg/csv2nifit.py \
#     --input_dir "/home/ahaas/data/1_simulated_data/$experiment_name" \
#     --threads 14 \
#     --scans $scans

# echo "Run manual domain adaptation"
# python vessel_graph_generation/domain_adaptation/domain_adaptation.py \
#     --input_dir "/home/ahaas/data/1_simulated_data/$experiment_name" \
#     --out_dir "/home/ahaas/data/2_manual_adapted_data" \
#     --threads 5 \
#     --scans $scans

echo "Copy masks to /home/ahaas/data/2_manual_adapted_data"
in_dir=/home/ahaas/data/1_simulated_data/$experiment_name
out_dir=/home/ahaas/data/2_manual_adapted_data/$experiment_name/masks
mkdir -p $out_dir
for mask_path in $in_dir/*; do
    cp $mask_path/airways/lung.nii.gz $out_dir/syn_$(basename $mask_path)_mask.nii.gz
done
