#!bin/bash
experiment_name=dataset_2023_11_15

python /home/ahaas/airway-seg/vessel_graph_generation/domain_adaptation/deformation.py \
    --input_dir /home/ahaas/data/2_manual_adapted_data/$experiment_name \
    --out_dir /home/ahaas/data/3_deformed_data/$experiment_name \
    --crop_foreground \
    --save_original \
    --threads 20
