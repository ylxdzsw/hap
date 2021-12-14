import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import fmoe

from torch.nn.parallel import DistributedDataParallel as DDP

rank=int(sys.argv[1])
world_size=4

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = str(world_size)
dist.init_process_group("nccl", rank=rank, world_size=world_size)

class SwitchTransformerEncoderLayer(nn.Module):
    def __init__(self):
        super(SwitchTransformerEncoderLayer, self).__init__()

        self.self_atten = torch.nn.MultiheadAttention(1024, 4, dropout=0.1)

        self.moe = fmoe.FMoETransformerMLP(
            num_expert=8, # this is the number of experts on *each* worker
            d_model=1024,
            d_hidden=4096,
            expert_rank=rank,
            top_k=1,
            world_size=world_size,
        )

        self.norm1 = torch.nn.LayerNorm(1024, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(1024, eps=1e-5)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self.moe(x))
        return x

    def _sa_block(self, x):
        x = self.self_atten(x, x, x, need_weights=False)[0]
        return self.dropout(x)

class NaiveSwitchTransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_atten = torch.nn.MultiheadAttention(1024, 4, dropout=0.1)

        self.moe = fmoe.FMoE(
            num_expert=8, # this is the number of experts on *each* worker
            d_model=1024,
            top_k=1,
            world_size=world_size,

            expert=lambda d: torch.nn.Sequential(
                LogShape(),
                torch.nn.Linear(d, 4096),
                torch.nn.ReLU(),
                torch.nn.Linear(4096, d),
            ),
        )

        self.norm1 = torch.nn.LayerNorm(1024, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(1024, eps=1e-5)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.norm1(x + self._sa_block(x))
        x = self.norm2(x + self.moe(x))
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
            torch.nn.TransformerEncoderLayer(1024, 4, 4096, 0.1)
            if i % 2 == 0 else
            # SwitchTransformerEncoderLayer()
            NaiveSwitchTransformerEncoderLayer()
            for i in range(4)
        ])

    def forward(self, x):
        for layer in self.layers:
            x=  layer(x)
        return torch.sum(x)

model = MoE().to(0)
model = fmoe.DistributedGroupedDataParallel(model)
# ddp_model = DDP(model, device_ids=[0])

optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
rand_input = torch.rand(4, 256, 1024).to(0) / 6

for i in range(10):
    start_time = time.time()
    loss = model(rand_input)
    dist.barrier(device_ids=[0])
    loss.backward()
    torch.cuda.synchronize()
    optimizer.step()
    dist.barrier()
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
