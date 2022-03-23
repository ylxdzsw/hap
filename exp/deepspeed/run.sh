cd "${BASH_SOURCE%/*}"

WORLDINFO=$(python -c 'import config;import json;print(json.dumps({f"{i}": [*range(config.ranks_per_card)] for i in range(config.world_size//config.ranks_per_card)}))')
echo $WORLDINFO

unset ALI

python -u -m deepspeed.launcher.launch \
    --world_info="$(base64 <<< "$WORLDINFO")" \
    --node_rank=$NODERANK \
    --master_addr=$(python -c 'import config;print(config.master_addr)') \
    --master_port=$(python -c 'import config;print(config.master_port)') \
    ds.py --deepspeed_config=ds_config.json
