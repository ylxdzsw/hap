import os
import sys
import math

rootpath = "/root/hidup"
sys.path.insert(1, f"{rootpath}/spmd")

model_name = "Tmlp"
# model_name = "Rmoe"
# model_name = "Rswitch"
# model_name = "Vmoe"
# model_name = "Vswitch"

world_size = 8
nlayers = 4
n_expert = 2 * world_size
batch_size = 32 * world_size
seqlen = 128
if model_name.startswith('V'):
    seqlen = 64
capacity_factor = 1.25
if model_name.endswith('moe'):
    capacity_factor *= 2
capacity = math.ceil(seqlen / n_expert * capacity_factor)
emsize = 768
nhid = emsize * 4

dropout = 0.1
nheads = 12

master_addr = "127.0.0.1"
master_port = 39262

trace = True
# trace = False

# use_hints = True
use_hints = False

report_per_iter_time = True
# report_per_iter_time = False

profile_noise = 0
# profile_noise = 0.8

lr = 5e-4

run_iter = 50
avg_iter = 20
log_iter = 10

# fp16 = True
fp16 = False

if os.environ.get("CPN", "") != "":
    cards_per_node = int(os.environ["CPN"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(cards_per_node)))
    offset = int(os.environ["NODERANK"]) * cards_per_node
    sys.argv.append(",".join(str(offset + i) for i in range(cards_per_node)))

def get_model(seed=None):
    import models

    if seed is not None:
        import torch
        torch.manual_seed(seed)

    if model_name == 'Tmlp':
        return models.TMLP(nhid=emsize, nlayers=nlayers)
    if model_name == 'Tmlp2':
        return models.TMLP2(nhid=emsize, nlayers=nlayers)
    if model_name == 'Ttransformer':
        return models.TTransformer(emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers)
    if model_name == 'Tmoe':
        return models.TMoE(emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers)

    if model_name == 'Rtransformer':
        ntokens, *_ = get_data()
        return models.RTransformer(ntokens=ntokens, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers)
    if model_name == 'Rmoe':
        ntokens, *_ = get_data()
        return models.RMoE(ntokens=ntokens, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers)
    if model_name == 'Rswitch':
        ntokens, *_ = get_data()
        return models.RSwitch(ntokens=ntokens, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers)

    if model_name == 'Vtransformer':
        nclasses, *_ = get_data()
        return models.VRransformer(nclasses=nclasses, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers)
    if model_name == 'Vmoe':
        nclasses, *_ = get_data()
        return models.VMoE(nclasses=nclasses, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers)
    if model_name == 'Vswitch':
        nclasses, *_ = get_data()
        return models.VSwitch(nclasses=nclasses, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers)

def get_data():
    if model_name.startswith('R'):
        return wikitext2()
        # return wikitext103()

    if model_name.startswith('V'):
        return cifar10()

    if model_name.startswith('T'):
        import torch
        x = torch.rand(batch_size, seqlen, emsize) / 6
        y = torch.rand(batch_size)
        def rep():
            while True:
                yield x, y
        return 0, rep()

def wikitext2():
    sys.path.insert(1, f"{rootpath}/wikitext")
    import data
    corpus = data.Corpus(f"{rootpath}/wikitext")
    train_data = data.segmentify(data.batchify(corpus.train, batch_size), seqlen)
    test_data = data.segmentify(data.batchify(corpus.test, batch_size), seqlen)
    valid_data = data.segmentify(data.batchify(corpus.valid, batch_size), seqlen)
    ntokens = world_size * (len(corpus.dictionary) // world_size + 1) # we have to ensure that it is dividable
    return ntokens, train_data, test_data, valid_data

def wikitext103():
    sys.path.insert(1, f"{rootpath}/wikitext")
    import data
    from pathlib import Path
    corpus = data.Corpus(f"{str(Path.home())}/.torchtext/cache/WikiText103/wikitext-103", is_103=True)
    train_data = data.segmentify(data.batchify(corpus.train, batch_size), seqlen)
    test_data = data.segmentify(data.batchify(corpus.test, batch_size), seqlen)
    valid_data = data.segmentify(data.batchify(corpus.valid, batch_size), seqlen)
    ntokens = world_size * (len(corpus.dictionary) // world_size + 1) # we have to ensure that it is dividable
    return ntokens, train_data, test_data, valid_data

def cifar10():
    import torch
    import torchvision
    def it(data):
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, drop_last=True)
        while True:
            yield from iter(loader)
    train_data = torchvision.datasets.CIFAR10(f"{rootpath}/cifar10", train=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10(f"{rootpath}/cifar10", train=False, transform=torchvision.transforms.ToTensor())
    return 10, it(train_data), it(test_data)

def input_shape():
    if model_name.startswith('R'):
        return { 'x': (batch_size, seqlen), 'y': (batch_size, seqlen) }
    if model_name.startswith('V'):
        return { 'x': (batch_size, 3, 32, 32), 'y': (batch_size,) }
    if model_name.startswith('T'):
        return { 'x': (batch_size, seqlen, emsize), 'y': (batch_size,) }

profiler_data = {
    "n_devices": world_size,

    "device_flops": 4139214925014,

    'all_gather': 1224592728, 'all_reduce': 611692856, 'reduce_scatter': 1130230706, 'all_to_all': 10701240728,
    # 'all_gather': 7429968322//2, 'all_reduce': 4421821530//2, 'reduce_scatter': 7537901123//2, 'all_to_all': 25451035500//2
}
