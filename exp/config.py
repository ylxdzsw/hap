import sys
sys.path.insert(1, f"{sys.path[0]}/../spmd")

world_size = 4
nhid = 4096
nlayers = 4
n_expert = 32
capacity = 10
batch_size = 128
seqlen = 256
emsize = 2048

dropout = 0.1
