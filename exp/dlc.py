import os
import re
import json

with open("config.py") as f:
    config = f.read()

config = re.sub(r'\nworld_size = .*', f"\nworld_size = {os.environ['WORLD_SIZE']}", config, count=1)
config = re.sub(r'\nmaster_addr = .*', f"\nmaster_addr = '{os.environ['MASTER_ADDR']}'", config, count=1)
config = re.sub(r'\nmaster_port = .*', f"\nmaster_port = {os.environ['MASTER_PORT']}", config, count=1)

if os.environ.get("NOARGV") == None:
    config += f"\nsys.argv.append('{os.environ['RANK']}')"

open("config.py", "w").write(config)

world_info = json.dumps({ f"{i}": [i] for i in range(int(os.environ['WORLD_SIZE'])) })
deepspeed = f"""
python -u -m deepspeed.launcher.launch \\
    --world_info=$(base64 <<< '{world_info}') \\
    --node_rank={os.environ['RANK']} \\
    --master_addr="{os.environ['MASTER_ADDR']}" \\
    --master_port="{os.environ['MASTER_PORT']}" \\
    ds.py --deepspeed_config=ds_config.json
"""

open("deepspeed/run.sh", "w").write(deepspeed)
