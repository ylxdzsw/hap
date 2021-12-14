import os

env = os.environ.copy()
env["PATH"] = "/home/swzhang/miniconda3/envs/fastmoe/bin:" + env["PATH"]
os.environ.update(env)

import deepspeed
deepspeed.init_distributed()
deepspeed.utils.groups.initialize()

import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn

import argparse
parser = argparse.ArgumentParser(description='fuk')
parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

class SwitchTransformerEncoderLayer(nn.Module):
    def __init__(self):
        super(SwitchTransformerEncoderLayer, self).__init__()

        self.self_atten = torch.nn.MultiheadAttention(1024, 4, dropout=0.1)

        self.moe = deepspeed.moe.layer.MoE(
            hidden_size=1024,
            expert=torch.nn.Sequential(
                torch.nn.Linear(1024, 4096),
                torch.nn.Linear(4096, 1024),
            ),
            num_experts=8,
            k=1,
            min_capacity=10,
        noisy_gate_policy='RSample')

        self.norm1 = torch.nn.LayerNorm(1024, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(1024, eps=1e-5)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self.moe(x)[0])
        return x

    def _sa_block(self, x):
        x = self.self_atten(x, x, x, need_weights=False)[0]
        return self.dropout(x)

class MoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(1024, 4, 4096, 0.1)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer()
            for i in range(4)
        ])

    def forward(self, x):
        for layer in self.layers:
            x=  layer(x)
        return torch.sum(x)

model = MoE()

model_engine, optimizer, _, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=filter(lambda p: p.requires_grad, model.parameters()))

# ddp_model = DDP(model, device_ids=[0])

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
rand_input = torch.rand(16, 256, 1024).to(model_engine.local_rank) / 6

for i in range(10):
    start_time = time.time()
    loss = model_engine(rand_input)
    model_engine.backward(loss)
    # torch.cuda.synchronize()
    model_engine.step()
    print(f"iter {i}, time {time.time() - start_time}s")

# from torch.profiler import profile, record_function, ProfilerActivity

# with profile(
#     activities= [ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule= torch.profiler.schedule(wait=1, warmup=1, active=4)
# ) as prof:
#     for _ in range(6):
#         with record_function("forward"):
#             loss = model(rand_input)
#         with record_function("backward"):
#             loss.backward()
#             torch.cuda.synchronize()
#         with record_function("update"):
#             optimizer.step()
#         dist.barrier()
#         prof.step()

# if rank == 0:
#     prof.export_chrome_trace("trace.json")

print(model(rand_input))
