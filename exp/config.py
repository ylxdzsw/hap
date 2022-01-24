import sys
sys.path.insert(1, f"{sys.path[0]}/../spmd")

model_name = "mlp"

world_size = 4
nlayers = 4
n_expert = 32
capacity = 10
batch_size = 32 * world_size
seqlen = 256
emsize = 1024
nhid = emsize * 4

dropout = 0.0
nheads = 4

master_addr = "127.0.0.1"
# master_addr = "10.28.1.24" # g9
master_port = 39261

trace = False

def get_model(seed=None):
    from models import MLP, MLP2, Transformer, MoE

    if seed is not None:
        import torch
        torch.manual_seed(seed)

    if model_name == 'mlp':
        return MLP(nhid=emsize, nlayers=nlayers)
    if model_name == 'mlp2':
        return MLP2(nhid=emsize, nlayers=nlayers)
    if model_name == 'transformer':
        return Transformer(emsize=emsize, nhead=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers)
    if model_name == 'moe':
        return MoE(emsize=emsize, nhead=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers)

profiler_data = {
    "n_devices": world_size,

    "device_flops": 5542786756118,

    # "device_flops": 4468888991664, # 1080Ti (g10), MLP
    # "device_flops": 2218545605794, # 1080Ti (g10), MoE


    # ==== NVLink (g11) ====
    "all_gather": 7703543732,
    "all_reduce": 4457607154,
    "reduce_scatter": 7724567251,
    "all_to_all": 21389930375,

    # ==== g9 g10 ====
    # "all_gather": 3502600835,
    # "all_reduce": 1888718528,
    # "reduce_scatter": 3722992647,
    # "all_to_all": 9616962998,
}
