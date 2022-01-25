import sys

if sys.path[0] == "":
    sys.path[0] = "."
sys.path.insert(1, f"{sys.path[0]}/../spmd")

model_name = "transformerR"

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

epoch = 40
log_iterval = 10

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

data_cache = None
def get_data():
    global data_cache
    if data_cache is None:
        sys.path.insert(1, f"{sys.path[0]}/../wikitext")
        import data
        corpus = data.Corpus(f"{sys.path[0]}/../wikitext")
        train_data = data.batchify(corpus.train, batch_size)
        test_data = data.batchify(corpus.test, batch_size)
        valid_data = data.batchify(corpus.valid, batch_size)
        data_cache = len(corpus.dictionary), train_data, test_data, valid_data
    return data_cache

def input_shape():
    if model_name.endswith('R'):
        return { 'x': (batch_size, seqlen), 'y': (batch_size, seqlen) }
    else:
        return { 'x': (batch_size, seqlen, emsize) }

profiler_data = {
    "n_devices": world_size,

    # "device_flops": 5603062517020, # V100 16G (g11), MLP
    "device_flops": 3312996495566, # V100 16G (g11), MoE
    # "device_flops": 4468888991664, # 1080Ti (g10), MLP
    # "device_flops": 2218545605794, # 1080Ti (g10), MoE

    "all_gather": 7586351942, "all_reduce": 4681009156, "reduce_scatter": 7900003407, "all_to_all": 21875592969, # NVLink (g11)
    # "all_gather": 3502600835, "all_reduce": 1888718528, "reduce_scatter": 3722992647, "all_to_all": 9616962998, # g9 g10
}
