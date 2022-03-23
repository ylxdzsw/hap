import os
import sys
import math

if sys.path[0] == "":
    sys.path[0] = "."
sys.path.insert(1, f"{sys.path[0]}/../spmd")

model_name = "moe"

world_size = 32
nlayers = 6
n_expert = 4 * world_size
batch_size = 32 * world_size
seqlen = 256
capacity_factor = 1.25
capacity = math.floor(seqlen / n_expert * capacity_factor)
emsize = 512
nhid = emsize * 4

dropout = 0.1
nheads = 4

# master_addr = "127.0.0.1"
# master_addr = "10.28.1.24" # g9
# master_addr = "10.28.1.27" # g12
master_addr = "172.26.161.155"
master_port = 39261

# trace = True
trace = False

epoch = 40
log_iterval = 10

profile_noise = 0
# profile_noise = 0.8

if os.environ.get("ALI", "") != "":
    ranks_per_card = int(os.environ["ALI"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(ranks_per_card)))
    offset = int(os.environ["NODERANK"]) * ranks_per_card
    sys.argv.append(",".join(str(offset + i) for i in range(ranks_per_card)))

def get_model(seed=None):
    from models import MLP, MLP2, Transformer, TransformerR, MoE

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

    if model_name == 'transformerR':
        ntokens, *_ = get_data()
        return TransformerR(ntokens=ntokens, seqlen=seqlen, emsize=emsize, nhead=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers)

def get_data():
    sys.path.insert(1, f"{sys.path[0]}/../wikitext")
    import data
    corpus = data.Corpus(f"{sys.path[0]}/../wikitext")
    train_data = data.batchify(corpus.train, batch_size)
    test_data = data.batchify(corpus.test, batch_size)
    valid_data = data.batchify(corpus.valid, batch_size)
    ntokens = world_size * (len(corpus.dictionary) // world_size + 1) # we have to ensure that it is dividable
    return ntokens, train_data, test_data, valid_data

def input_shape():
    if model_name.endswith('R'):
        return { 'x': (batch_size, seqlen), 'y': (batch_size, seqlen) }
    else:
        return { 'x': (batch_size, seqlen, emsize) }

profiler_data = {
    "n_devices": world_size,

    # "device_flops": 5603062517020, # V100 16G (g11), MLP
    # "device_flops": 3312996495566, # V100 16G (g11), MoE
    # "device_flops": 4468888991664, # 1080Ti (g10), MLP
    # "device_flops": 2218545605794, # 1080Ti (g10), MoE
    # "device_flops": 2474540597913, # A10 (DLC), MoE
    "device_flops": 4139214925014, # V100 16G (ali), MoE


    # 'all_gather': 1629540629, 'all_reduce': 770636359, 'reduce_scatter': 1568092051, 'all_to_all': 5875506734, # four cards one per machine
    # 'all_gather': 1444972440, 'all_reduce': 644687571, 'reduce_scatter': 1409464500, 'all_to_all': 9295475658, # 32 cards on 4 machines
    'all_gather': 1138061790, 'all_reduce': 553212015, 'reduce_scatter': 1129505905, 'all_to_all': 10178889989, # 64 cards on 8 machines

    # "all_gather": 7586351942, "all_reduce": 4681009156, "reduce_scatter": 7900003407, "all_to_all": 21875592969, # NVLink (g11)
    # "all_gather": 3502600835, "all_reduce": 1888718528, "reduce_scatter": 3722992647, "all_to_all": 9616962998, # g9 g10
}
