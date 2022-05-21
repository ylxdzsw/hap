import config

import os
import sys
import math
import datetime
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from contextlib import nullcontext
import numpy as np

from models import positional_encoding, PatchEmbed, append_cls_token, get_cls_token

from utils import *
import fmoe

class SwitchTransformerEncoderLayer(nn.Module):
    def __init__(self, global_rank):
        super().__init__()

        self.self_atten = torch.nn.MultiheadAttention(config.emsize, config.nheads, dropout=config.dropout, batch_first=True)

        self.moe = fmoe.FMoETransformerMLP(
            num_expert=config.n_expert // config.world_size, # this is the number of experts on *each* worker
            d_model=config.emsize,
            d_hidden=config.nhid,
            expert_rank=global_rank,
            top_k=1,
            world_size=config.world_size,
        )

        self.norm1 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x, src_mask = None):
        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm2(x + self.moe(x))
        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_atten(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout(x)

class Top2TransformerEncoderLayer(nn.Module):
    def __init__(self, global_rank):
        super().__init__()

        self.self_atten = torch.nn.MultiheadAttention(config.emsize, config.nheads, dropout=config.dropout, batch_first=True)

        self.moe = fmoe.FMoETransformerMLP(
            num_expert=config.n_expert // config.world_size, # this is the number of experts on *each* worker
            d_model=config.emsize,
            d_hidden=config.nhid,
            expert_rank=global_rank,
            top_k=2,
            world_size=config.world_size,
        )

        self.norm1 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.dropout = torch.nn.Dropout(config.dropout)

    def forward(self, x, src_mask = None):
        x = self.norm1(x + self._sa_block(x, src_mask))
        x = self.norm2(x + self.moe(x))
        return x

    def _sa_block(self, x, attn_mask):
        x = self.self_atten(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout(x)

class NaiveSwitchTransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        self.self_atten = torch.nn.MultiheadAttention(config.emsize, config.nheads, dropout=config.dropout, batch_first=True)

        self.moe = fmoe.FMoE(
            num_expert=config.n_expert // config.world_size, # this is the number of experts on *each* worker
            d_model=config.emsize,
            top_k=1,
            world_size=config.world_size,

            expert=lambda d: torch.nn.Sequential(
                # LogShape(),
                torch.nn.Linear(d, config.nhid),
                torch.nn.ReLU(),
                torch.nn.Linear(config.nhid, d),
            ),
        )

        self.norm1 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.norm2 = torch.nn.LayerNorm(config.emsize, eps=1e-5)
        self.dropout = torch.nn.Dropout(config.dropout)

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

class TMoE(torch.nn.Module):
    def __init__(self, global_rank) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer(global_rank)
            # NaiveSwitchTransformerEncoderLayer()
            for i in range(config.nlayers)
        ])

    def forward(self, x, y):
        for layer in self.layers:
            x = layer(x)
        return torch.sum(x)

class RMoE(torch.nn.Module):
    def __init__(self, ntokens, global_rank):
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
            Top2TransformerEncoderLayer(global_rank)
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
    def __init__(self, ntokens, global_rank):
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
            SwitchTransformerEncoderLayer(global_rank)
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
    def __init__(self, nclasses, global_rank):
        super().__init__()
        self.emsize = config.emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=config.emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, config.emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, config.seqlen + 1, config.emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            Top2TransformerEncoderLayer(global_rank)
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
    def __init__(self, nclasses, global_rank):
        super().__init__()
        self.emsize = config.emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=config.emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, config.emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, config.seqlen + 1, config.emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(config.emsize, config.nheads, config.nhid, config.dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer(global_rank)
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

def run(global_rank, local_rank):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank, timeout=datetime.timedelta(hours=2))

    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)

    ntokens, train_data, *_ = config.get_data()

    M = { "Rmoe": RMoE, "Rswitch": RSwitch, "Vmoe": VMoE, "Vswitch": VSwitch }[config.model_name]

    model = M(ntokens, global_rank).to(local_rank)
    model = fmoe.DistributedGroupedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    result_times = []
    strat_time = last_iter_time = time.time()
    total_loss = 0
    for iter in range(config.run_iter):
        optimizer.zero_grad()
        x, y = next(train_data)
        x = x.chunk(config.world_size, 0)[global_rank].cuda(local_rank)
        y = y.chunk(config.world_size, 0)[global_rank].cuda(local_rank)

        with torch.autocast(device_type="cuda") if config.fp16 else nullcontext() :
            loss = model(x, y)

        aggregated_loss = loss.detach().clone()
        dist.reduce(aggregated_loss, 0)
        if global_rank == 0:
            total_loss += aggregated_loss.cpu().numpy() / config.batch_size / config.seqlen
            if iter % config.log_iter == 0:
                print(f"loss (log ppl) {iter}: {total_loss / config.log_iter:.3f}, wall clock: {time.time() - strat_time:.3f}")
                total_loss = 0
        # dist.barrier(device_ids=[rank])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        # torch.cuda.synchronize()
        optimizer.step()
        # dist.barrier()
        if config.report_per_iter_time and local_rank == 0:
            iter_duration = time.time() - last_iter_time
            result_times.append(iter_duration)
            last_iter_time += iter_duration
            print("iter time: ", iter_duration)
            print("avg±std:", np.mean(result_times[-config.avg_iter:]), np.std(result_times[-config.avg_iter:]))

    if not config.trace:
        return

    x, y = next(train_data)
    x = x.chunk(config.world_size, 0)[global_rank].cuda(local_rank)
    y = y.chunk(config.world_size, 0)[global_rank].cuda(local_rank)
    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes = True,
        # profile_memory = True,
        schedule = torch.profiler.schedule(wait=1, warmup=10, active=4)
    ) as prof:
        for _ in range(15):
            with record_function("forward"):
                with torch.autocast(device_type="cuda") if config.fp16 else nullcontext() :
                    loss = model(x, y)
            with record_function("backward"):
                loss.backward()
                torch.cuda.synchronize()
            with record_function("update"):
                optimizer.step()
            dist.barrier()
            prof.step()

    if global_rank == 0:
        # print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json")

if __name__ == '__main__':
    ranks = [ int(x) for x in sys.argv[1].split(',') ]

    if torch.cuda.device_count() != len(ranks):
        print("forget to set CUDA_VISIBLE_DEVICES")
        raise SystemExit

    import os
    os.environ['MASTER_ADDR'] = str(config.master_addr)
    os.environ['MASTER_PORT'] = str(config.master_port)
    os.environ['WORLD_SIZE'] = str(config.world_size)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    for local_rank, global_rank in enumerate(ranks):
        mp.Process(target=run, args=(global_rank, local_rank)).start()

    for p in mp.active_children():
        p.join()
