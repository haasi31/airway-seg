experiment_name=dataset_2023_11_14_one
num_samples=1

echo "Run vessel graph simulation"
python /home/ahaas/airway-seg/generate_vessel_graph.py \
    --config_file /home/ahaas/airway-seg/vessel_graph_generation/configs/vessels_config.yml \
    --num_samples $num_samples \
    --threads 14 \
    --experiment_name $experiment_name

