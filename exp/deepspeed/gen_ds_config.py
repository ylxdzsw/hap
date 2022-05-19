import config
import json

base = {
    "train_micro_batch_size_per_gpu": config.batch_size // config.world_size,
    "steps_per_print": 2000,
    "optimizer": {
      "type": "Adam",
      "params": {
        "lr": 0.0001,
        "betas": [
          0.8,
          0.999
        ],
        "eps": 1e-8,
        "weight_decay": 3e-7
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 0.001,
        "warmup_num_steps": 1000
      }
    },
    "gradient_clipping": 1.0,
    "prescale_gradients": False,
    "wall_clock_breakdown": False
}

if config.fp16:
    base["fp16"] = {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 32,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }

print(json.dumps(base))
