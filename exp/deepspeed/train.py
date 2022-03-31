import config

import os
import datetime

env = os.environ.copy()
env["PATH"] = "/home/swzhang/miniconda3/envs/th19/bin:" + env["PATH"]
os.environ.update(env)

import deepspeed
deepspeed.init_distributed(timeout=datetime.timedelta(hours=2))
deepspeed.utils.groups.initialize(ep_size=config.world_size)

import math
import time
import torch
import torch.distributed as dist
import torch.nn as nn

from models import positional_encoding, PatchEmbed, append_cls_token, get_cls_token

import argparse
parser = argparse.ArgumentParser(description='asd')
parser.add_argument('--local_rank',
                    type=int,
                    default=-1,
                    help='local rank passed from distributed launcher')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

class Top2TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

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
            k=2,
            capacity_factor=config.capacity_factor / 2, # our capacity factor is multiplied by k
            noisy_gate_policy='RSample'
        )

        self.norm1 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x, src_mask = None):
        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm2(x + self.moe(x)[0])
        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_atten(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout(x)

class SwitchTransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

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

    def forward(self, x, src_mask = None):
        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm2(x + self.moe(x)[0])
        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_atten(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout(x)

class LogShape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x

class TMoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer()
            for i in range(config.nlayers)
        ])

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
        return torch.sum(x)

class RMoE(torch.nn.Module):
    def __init__(self, ntokens):
        super().__init__()
        self.emsize = config.emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.register_buffer('pe', torch.Tensor(positional_encoding(config.seqlen, config.emsize)).unsqueeze(0))
        # self.pe_dropout = torch.nn.Dropout(dropout)
        self.register_buffer('src_mask', torch.triu(torch.full((config.seqlen, config.seqlen), float('-inf')), diagonal=1))
        self.encoder = torch.nn.Embedding(ntokens, config.emsize)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            Top2TransformerEncoderLayer()
            for i in range(config.nlayers)
        ])
        self.decoder = torch.nn.Linear(config.emsize, ntokens)

    def forward(self, x, y):
        x = self.encoder(x) * math.sqrt(self.emsize)
        src_mask = self.src_mask # otherwise it produces a load statement each time, which not only makes multiple communications but also bug in compiling (we will slice the same tensor multiple times)
        x += self.pe
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x.transpose(1, 2), y) # the input to NLL loss is (N, C, ...), so we move the class prediction to the second dimension

class RSwitch(torch.nn.Module):
    def __init__(self, ntokens):
        super().__init__()
        self.emsize = config.emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.register_buffer('pe', torch.Tensor(positional_encoding(config.seqlen, config.emsize)).unsqueeze(0))
        # self.pe_dropout = torch.nn.Dropout(dropout)
        self.register_buffer('src_mask', torch.triu(torch.full((config.seqlen, config.seqlen), float('-inf')), diagonal=1))
        self.encoder = torch.nn.Embedding(ntokens, config.emsize)
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer()
            for i in range(config.nlayers)
        ])
        self.decoder = torch.nn.Linear(config.emsize, ntokens)

    def forward(self, x, y):
        x = self.encoder(x) * math.sqrt(self.emsize)
        src_mask = self.src_mask # otherwise it produces a load statement each time, which not only makes multiple communications but also bug in compiling (we will slice the same tensor multiple times)
        x += self.pe
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x.transpose(1, 2), y) # the input to NLL loss is (N, C, ...), so we move the class prediction to the second dimension

class VMoE(torch.nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.emsize = config.emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=config.emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, config.emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, config.seqlen + 1, config.emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            Top2TransformerEncoderLayer()
            for i in range(config.nlayers)
        ])
        self.decoder = torch.nn.Linear(config.emsize, nclasses)

    def forward(self, x, y):
        x = self.patch_embed(x)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = get_cls_token(x) # embedding of the class token
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)

class VSwitch(torch.nn.Module):
    def __init__(self, nclasses):
        super().__init__()
        self.emsize = config.emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=config.emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, config.emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, config.seqlen + 1, config.emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer()
            for i in range(config.nlayers)
        ])
        self.decoder = torch.nn.Linear(config.emsize, nclasses)

    def forward(self, x, y):
        x = self.patch_embed(x)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = get_cls_token(x) # embedding of the class token
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)

ntokens, train_data, *_ = config.get_data()

model = { "Rmoe": RMoE, "Rswitch": RSwitch, "Vmoe": VMoE, "Vswitch": VSwitch }[config.model_name](ntokens)

model_engine, optimizer, _, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=filter(lambda p: p.requires_grad, model.parameters()))

result_times = []

for i in range(100):
    x, y = next(train_data)
    x = x.chunk(config.world_size, 0)[dist.get_rank()].cuda(model_engine.local_rank)
    y = y.chunk(config.world_size, 0)[dist.get_rank()].cuda(model_engine.local_rank)
    start_time = time.time()
    loss = model_engine(x, y)
    model_engine.backward(loss)
    # torch.cuda.synchronize()
    model_engine.step()
    if model_engine.local_rank == 0:
        print(f"iter {i}, time {time.time() - start_time}s")
        result_times.append(time.time() - start_time)
        print("avg:", sum(result_times[-50:]) / len(result_times[-50:]))

if not config.trace:
    raise SystemExit

from torch.profiler import profile, record_function, ProfilerActivity

x, y = next(train_data)
x = x.chunk(config.world_size, 0)[dist.get_rank()].cuda(model_engine.local_rank)
y = y.chunk(config.world_size, 0)[dist.get_rank()].cuda(model_engine.local_rank)
with profile(
    activities= [ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule = torch.profiler.schedule(wait=1, warmup=10, active=4)
) as prof:
    for _ in range(15):
        with record_function("forward"):
            loss = model_engine(x, y)
        with record_function("backward"):
            model_engine.backward(loss)
            # torch.cuda.synchronize()
        with record_function("update"):
            model_engine.step()
        # dist.barrier()
        prof.step()

if model_engine.local_rank == 0:
    prof.export_chrome_trace("trace.json")

print(model(x, y))
