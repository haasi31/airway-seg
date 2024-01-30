experiment_name=dataset_2023_11_15
scans="010 027 221"

echo "Run manual domain adaptation"
python vessel_graph_generation/domain_adaptation/domain_adaptation.py \
    --input_dir "/home/ahaas/data/1_simulated_data/$experiment_name" \
    --out_dir "/home/ahaas/data/2_manual_adapted_data" \
    --threads 5 \
    --scans $scans