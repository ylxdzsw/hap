from __future__ import annotations

import math
import torch
import torch.fx
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, nhid=2048, nlayers=10):
        super().__init__()
        modlist = []
        for _ in range(nlayers):
            modlist.append(torch.nn.Linear(nhid, nhid))
            modlist.append(torch.nn.Sigmoid())
        self.layers = torch.nn.ModuleList(modlist)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sum(x)

class MLP2(torch.nn.Module):
    def __init__(self, nhid=2048, nlayers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(nhid, nhid) for _ in range(nlayers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = torch.bmm(x, x)
            x = torch.bmm(x, x)
        return torch.sum(x)

class Transformer(torch.nn.Module):
    def __init__(self, emsize=2048, nhead=4, nhid=2048, dropout=0.2, nlayers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([ torch.nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True) for _ in range(nlayers) ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sum(x)

class MoE(torch.nn.Module):
    def __init__(self, emsize, nhead, nhid, dropout, n_expert, capacity, nlayers=2): # capacity should be seq_len / n_expert * factor
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True)
            if i % 2 == 0 else
            SwitchTransformerEncoderLayer(emsize, nhead, nhid, dropout, n_expert=n_expert, capacity=capacity)
            for i in range(nlayers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.sum(x)

class TransformerR(torch.nn.Module):
    def __init__(self, ntokens, seqlen, emsize=2048, nhead=4, nhid=2048, dropout=0.2, nlayers=2):
        super().__init__()
        self.emsize = emsize
        self.criterion = torch.nn.NLLLoss(reduction='sum')
        self.register_buffer('pe', torch.Tensor(positional_encoding(seqlen, emsize)).unsqueeze(0))
        # self.pe_dropout = torch.nn.Dropout(dropout)
        self.register_buffer('src_mask', torch.triu(torch.full((seqlen, seqlen), float('-inf')), diagonal=1))
        self.encoder = torch.nn.Embedding(ntokens, emsize)
        self.layers = torch.nn.ModuleList([ torch.nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout, batch_first=True) for _ in range(nlayers) ])
        self.decoder = torch.nn.Linear(emsize, ntokens)

    def forward(self, x, y):
        x = self.encoder(x) * math.sqrt(self.emsize)
        src_mask = self.src_mask # otherwise it produces a load statement each time, which not only makes multiple communications but also bug in compiling (we will slice the same tensor multiple times)
        x += self.pe
        for layer in self.layers:
            x = layer(x, src_mask)
        x = self.decoder(x)
        x = torch.log_softmax(x, dim=-1)
        return self.criterion(x.transpose(1, 2), y) # the input to NLL loss is (N, C, ...), so we move the class prediction to the second dimension

class SwitchTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, d_hidden=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, n_expert=4, capacity=None) -> None:
        super().__init__()

        self.self_attn = torch.nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.gate_weight = torch.nn.Parameter(torch.empty((d_model, n_expert)))
        torch.nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

        self.w1 = torch.nn.Parameter(torch.empty((n_expert, d_model, d_hidden)))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))

        self.dropout = torch.nn.Dropout(dropout)

        self.w2 = torch.nn.Parameter(torch.empty((n_expert, d_hidden, d_model)))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.n_expert = n_expert
        self.capacity = capacity
        self.activation = activation

    def forward(self, src: torch.nn.Tensor, src_mask: torch.nn.Tensor | None = None, src_key_padding_mask: torch.nn.Tensor | None = None) -> torch.nn.Tensor:
        """
        gate_input: (batch, seq_len, d_model)
        dispatch_tensor: (batch, seq_len, n_expert, capacity)
        expert_inputs: (batch, n_expert, capacity, d_model)
        expert_outputs: (batch, n_expert, capacity, d_model)
        combine_tensor: (batch, seq_len, n_expert, capacity)
        outputs: (batch, seq_len, d_model)
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._moe_block(x))

        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    def _moe_block(self, x):
        dispatch_tensor, combine_tensor = switch_gating(x, self.n_expert, self.capacity, self.gate_weight, train=True) # (batch, seq_len, n_expert, capacity)

        expert_inputs = torch.einsum("bsd,bsec->becd", x, dispatch_tensor) # (batch, n_expert, capacity, d_model)

        h = torch.einsum("edh,becd->bech", self.w1, expert_inputs)

        h = self.activation(h)

        expert_outputs = torch.einsum("ehd,bech->becd", self.w2, h)

        output = torch.einsum("becd,bsec->bsd", expert_outputs, combine_tensor)

        return output

@torch.fx.wrap
def switch_gating(gate_input, n_expert, capacity, gate_weight, train: bool = True):
    return _switch_gating(gate_input, n_expert, capacity, gate_weight, train)

# @torch.jit.script
def _switch_gating(gate_input, n_expert: int, capacity: int, gate_weight, train: bool = True):
    gate_logits = torch.matmul(gate_input, gate_weight) # (batch, seq_len, n_expert)
    raw_gates = F.softmax(gate_logits, dim=2) # (batch, seq_len, n_expert)

    expert_gate, expert_index = torch.topk(raw_gates, k=1, dim=2, largest=True) # (batch, seq_len, 1)
    expert_gate = torch.squeeze(expert_gate, dim=2) # (batch, seq_len)
    expert_index = torch.squeeze(expert_index, dim=2) # (batch, seq_len)

    expert_mask = F.one_hot(expert_index, n_expert) # (batch, seq_len, n_expert)

    position_in_expert = torch.cumsum(expert_mask, dim=1) * expert_mask # (batch, seqlen, n_expert)
    expert_mask *= position_in_expert < capacity # (batch, seq_len, n_expert)
    position_in_expert *= position_in_expert < capacity # (batch, seq_len, n_expert)

    expert_mask_flat = torch.sum(expert_mask, dim=2, keepdim=False) # (batch, seq_len)

    combine_tensor = ( # (batch, seq_len, n_expert, capacity)
        torch.unsqueeze(torch.unsqueeze(expert_gate * expert_mask_flat, 2), 3) * # (batch, seq_len, 1, 1)
        torch.unsqueeze(F.one_hot(expert_index, n_expert), 3) * # (batch, seq_len, n_expert, 1) # TODO: why not use expert_mask?
        F.one_hot(position_in_expert, capacity)) # (batch, seq_len, n_expert, capacity)

    dispatch_tensor = (combine_tensor > 0).to(torch.float32)

    return dispatch_tensor, combine_tensor

def positional_encoding(seqlen, emsize):
    import numpy as np
    p = np.array([[pos / np.power(10000, 2 * (j // 2) / emsize) for j in range(emsize)] for pos in range(seqlen)])
    p[:, 0::2] = np.sin(p[:, 0::2])
    p[:, 1::2] = np.cos(p[:, 1::2])
    return p




# class PatchEmbedding(nn.Module):
#     """ 2D Image to Patch Embedding
#     """
#     def __init__(self,
#                  img_size: int,
#                  patch_size: int,
#                  in_chans: int,
#                  embed_size: int,
#                  dtype: dtype = None,
#                  flatten: bool = True,
#                  weight_initializer: Callable = init.kaiming_uniform_(a=math.sqrt(5)),
#                  bias_initializer: Callable = init.xavier_uniform_(a=1, scale=1),
#                  position_embed_initializer: Callable = init.zeros_()):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
#         self.num_patches = self.grid_size[0] * self.grid_size[1]
#         self.flatten = flatten

#         self.weight = nn.Parameter(
#             torch.empty((embed_size, in_chans, *self.patch_size), device=get_current_device(), dtype=dtype))
#         self.bias = nn.Parameter(torch.empty(embed_size, device=get_current_device(), dtype=dtype))
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_size))

#         self.reset_parameters(weight_initializer, bias_initializer, position_embed_initializer)

#     def reset_parameters(self, weight_initializer, bias_initializer, position_embed_initializer):
#         fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
#         weight_initializer(self.weight, fan_in=fan_in, fan_out=fan_out)
#         bias_initializer(self.bias, fan_in=fan_in)
#         position_embed_initializer(self.pos_embed)

#     def forward(self, input_: Tensor) -> Tensor:
#         B, C, H, W = input_.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         output = F.conv2d(input_, self.weight, self.bias, stride=self.patch_size)
#         if self.flatten:
#             output = output.flatten(2).transpose(1, 2)  # BCHW -> BNC

#         cls_token = self.cls_token.expand(output.shape[0], -1, -1)
#         output = torch.cat((cls_token, output), dim=1)
#         output = output + self.pos_embed
#         return output
