experiment_name=dataset_2023_11_15

echo "Run voxelization"
python /home/ahaas/airway-seg/csv2nifit.py \
    --input_dir "/home/ahaas/data/1_simulated_data/$experiment_name" \
    --threads 14 \
    --min_radius 0.0011 \
    #--scans "010" #"027" "221"