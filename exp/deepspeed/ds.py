import config

import os
import datetime

env = os.environ.copy()
env["PATH"] = "/home/swzhang/miniconda3/envs/th19/bin:" + env["PATH"]
os.environ.update(env)

import deepspeed
deepspeed.init_distributed(timeout=datetime.timedelta(hours=2))
deepspeed.utils.groups.initialize(ep_size=config.world_size)

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

        self.self_atten = torch.nn.MultiheadAttention(config.emsize, config.nheads, dropout=config.dropout, batch_first=True)

        self.moe = deepspeed.moe.layer.MoE(
            hidden_size=config.emsize,
            expert=torch.nn.Sequential(
                # LogShape(),
                torch.nn.Linear(config.emsize, config.nhid),
                torch.nn.ReLU(),
                torch.nn.Linear(config.nhid, config.emsize),
            ),
            num_experts=config.n_expert,
            k=1,
            capacity_factor=config.capacity_factor,
            noisy_gate_policy='RSample'
        )

        self.norm1 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self.moe(x)[0])
        return x

    def _sa_block(self, x):
        x = self.self_atten(x, x, x, need_weights=False)[0]
        return self.dropout(x)

class LogShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x

class MoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer()
            for i in range(config.nlayers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x=  layer(x)
        return torch.sum(x)

model = MoE()

model_engine, optimizer, _, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=filter(lambda p: p.requires_grad, model.parameters()))

result_times = []

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
rand_input = torch.rand(config.batch_size // config.world_size, config.seqlen, config.emsize).to(model_engine.local_rank) / 6

for i in range(100):
    start_time = time.time()
    loss = model_engine(rand_input)
    model_engine.backward(loss)
    # torch.cuda.synchronize()
    model_engine.step()
    if model_engine.local_rank == 0:
        print(f"iter {i}, time {time.time() - start_time}s")
        result_times.append(time.time() - start_time)

if model_engine.local_rank == 0:
    print("avg:", sum(result_times[-50:]) / 50)

if not config.trace:
    raise SystemExit

from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities= [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule = torch.profiler.schedule(wait=1, warmup=10, active=4)
) as prof:
    for _ in range(15):
        with record_function("forward"):
            loss = model_engine(rand_input)
        with record_function("backward"):
            model_engine.backward(loss)
            # torch.cuda.synchronize()
        with record_function("update"):
            model_engine.step()
        # dist.barrier()
        prof.step()

if model_engine.local_rank == 0:
    prof.export_chrome_trace("trace.json")

print(model(rand_input))
