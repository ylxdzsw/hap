echo "Remember to check this file and ds_config.json. This script does not automatically read all config."
python -u -m deepspeed.launcher.launch \
    --world_info=$(base64 <<< '{"10.28.1.27": [0, 1, 2, 3], "10.28.1.28": [0, 1, 2, 3]}') \
    --node_rank=$1 \
    --master_addr=$(python -c 'import config;print(config.master_addr)') \
    --master_port=$(python -c 'import config;print(config.master_port)') \
    ds.py --deepspeed_config=ds_config.json
