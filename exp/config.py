import sys
sys.path.insert(1, f"{sys.path[0]}/../spmd")

world_size = 8
nlayers = 6
n_expert = 32
capacity = 10
batch_size = 128
seqlen = 256
emsize = 1024
nhid = emsize * 4

dropout = 0.1
nheads = 4

master_addr = "10.28.1.26"
master_port = 39261
