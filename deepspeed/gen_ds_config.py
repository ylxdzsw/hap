import config
import json

base = {
    "train_micro_batch_size_per_gpu": config.batch_size // config.world_size,
    "steps_per_print": 2000000,
    "optimizer": {
      "type": "Adam",
      "params": {
        "lr": config.lr
      }
    },
    "gradient_clipping": 0.5,
    # "zero_optimization": {
    #     "stage": 2,
    # }
}

# if config.fp16:
#     base["fp16"] = {
#         "enabled": True,
#         "loss_scale": 0,
#         "initial_scale_power": 32,
#         "loss_scale_window": 1000,
#         "hysteresis": 2,
#         "min_loss_scale": 1
#     }

print(json.dumps(base))
