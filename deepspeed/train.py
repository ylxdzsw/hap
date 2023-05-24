import config

import os
import datetime
import numpy as np

# env = os.environ.copy()
# env["PATH"] = "/home/swzhang/miniconda3/envs/th19/bin:" + env["PATH"]
# os.environ.update(env)

import deepspeed
deepspeed.init_distributed(timeout=datetime.timedelta(hours=2))
# deepspeed.utils.groups.initialize(ep_size=config.world_size)

import math
import time
import torch
import torch.distributed as dist
import torch.nn as nn

from models import positional_encoding, PatchEmbed, append_cls_token, get_cls_token, new_segment

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
            ep_size=config.world_size,
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
            ep_size=config.world_size,
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

class VVGG(torch.nn.Module):
    def __init__(self, nclasses, dropout=config.dropout, segmentation=False):
        super().__init__()

        def make_layers(cfg):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == "M":
                    layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                    in_channels = v
            return torch.nn.Sequential(*layers)

        self.features = make_layers([64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"])
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(4096, nclasses),
        )
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.segmentation = segmentation

    def forward(self, x, y):
        for layer in self.features:
            x = layer(x)
            if self.segmentation and isinstance(layer, torch.nn.MaxPool2d):
                x = new_segment(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)

class VTransformer(torch.nn.Module):
    def __init__(self, nclasses, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=True):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')

        assert seqlen == 8 * 8

        self.patch_embed = PatchEmbed((32, 32), (4, 4), embed_dim=emsize)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emsize))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, seqlen + 1, emsize)) # seqlen patches + 1 cls token

        self.layers = torch.nn.ModuleList([ torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True) for _ in range(nlayers) ])
        self.decoder = torch.nn.Linear(emsize, nclasses)
        self.segmentation = segmentation

    def forward(self, x, y):
        # x: N, 3, 32, 32
        x = self.patch_embed(x)
        x = append_cls_token(x, self.cls_token)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)
            if self.segmentation:
                x = new_segment(x)

        x = get_cls_token(x) # embedding of the class token
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x, y)

class RTransformer(torch.nn.Module):
    def __init__(self, ntokens, seqlen=config.seqlen, emsize=config.emsize, nheads=config.nheads, nhid=config.nhid, dropout=config.dropout, nlayers=config.nlayers, segmentation=True):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.register_buffer('pe', torch.Tensor(positional_encoding(seqlen, emsize)).unsqueeze(0))
        # self.pe_dropout = torch.nn.Dropout(dropout)
        self.register_buffer('src_mask', torch.triu(torch.full((seqlen, seqlen), float('-inf')), diagonal=1))
        self.encoder = torch.nn.Embedding(ntokens, emsize)
        self.layers = torch.nn.ModuleList([ torch.nn.TransformerEncoderLayer(emsize, nheads, nhid, dropout, batch_first=True) for _ in range(nlayers) ])
        self.decoder = torch.nn.Linear(emsize, ntokens)
        self.segmentation = segmentation

    def forward(self, x, y):
        x = self.encoder(x) * math.sqrt(self.emsize)
        src_mask = self.src_mask # otherwise it produces a load statement each time, which not only makes multiple communications but also bug in compiling (we will slice the same tensor multiple times)
        x += self.pe
        for layer in self.layers:
            x = layer(x, src_mask)
            if self.segmentation:
                x = new_segment(x)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x.transpose(1, 2), y) # the input to NLL loss is (N, C, ...), so we move the class prediction to the second dimension


ntokens, train_data, *_ = config.get_data()

model = { "Rmoe": RMoE, "Rswitch": RSwitch, "Vmoe": VMoE, "Vswitch": VSwitch, "Vvgg": VVGG, "Vtransformer": VTransformer, "Rtransformer": RTransformer }[config.model_name](ntokens)

model_engine, optimizer, _, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=filter(lambda p: p.requires_grad, model.parameters()))

global_rank = dist.get_rank()

result_times = []
strat_time = last_iter_time = time.time()
total_loss = 0
for iter in range(config.run_iter):
    x, y = next(train_data)
    x = x.chunk(config.world_size, 0)[global_rank].cuda(model_engine.local_rank)
    y = y.chunk(config.world_size, 0)[global_rank].cuda(model_engine.local_rank)

    loss = model_engine(x, y)
    aggregated_loss = loss.detach().clone()
    dist.reduce(aggregated_loss, 0)

    if global_rank == 0:
        total_loss += aggregated_loss.cpu().numpy() / config.batch_size / config.seqlen
        if iter % config.log_iter == 0:
            print(f"loss (log ppl) {iter}: {total_loss / config.log_iter:.3f}, wall clock: {time.time() - strat_time:.3f}")
            total_loss = 0

    model_engine.backward(loss)
    # torch.cuda.synchronize()
    model_engine.step()

    if config.report_per_iter_time and model_engine.local_rank == 0:
        iter_duration = time.time() - last_iter_time
        result_times.append(iter_duration)
        last_iter_time += iter_duration
        print("iter time: ", iter_duration)
        print("avg±std:", np.mean(result_times[-config.avg_iter:]), np.std(result_times[-config.avg_iter:]))

if not config.trace:
    raise SystemExit

from torch.profiler import profile, record_function, ProfilerActivity

x, y = next(train_data)
x = x.chunk(config.world_size, 0)[global_rank].cuda(model_engine.local_rank)
y = y.chunk(config.world_size, 0)[global_rank].cuda(model_engine.local_rank)
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
