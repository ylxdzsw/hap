HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis


## Docker Image

We provide a Docker image with all dependencies pre-installed and pre-compiled for artifact evaluation:
https://hub.docker.com/r/ylxdzsw/hap. It can be downloaded with `docker pull ylxdzsw/hap:ae`.


## Build From Source

HAP is partially implemented in Rust and requires a Rust compiler to build. The building toolchain can be installed by
following https://rustup.rs. Currently HAP requires a nightly build of the Rust compiler.

Run the following command to build HAP:

```
cargo build --release
```


## Installing Python Dependencies

HAP is implemented on top of PyTorch 1.13.1. It can be installed, along with its own dependencies, by following
https://pytorch.org/get-started/previous-versions/#v1131.

Alternatively, we provides an environment file `environment.yml` for reproducing the experiment environment with conda
(https://conda.io).


## File Structures

Important files include:

- hap.rs: the main source file of HAP. Inside it, `load_fx_graph` parses a PyTorch fx graph into an internal graph
representation, which can be analyzed with `analyze_rgraph` to build the Hoare triples. `a_star` then synthesizes a
program with the Hoare triples with given sharding ratios (Eq. (1) in Sec. 3.1). `sharding_ratio_optimization` uses
Coin-or CBC optimizer to solve the optimization problem (Eq. (3) in Sec. 5.1) for the synthesized program.

- collectives.py: implementation of the collective operators for heterogeneous clusters, including the different
implementations of `All-Gather` (Sec. 2.5.1).

- config.py: the configurations of HAP. It needs to be frequently edited during experimentation.

- profiler.py: the script for profiling device flops and collective communication bandwidths. When run without rank
arguments (e.g. `python profiler.py`), it runs the model on a single card and calculates the device flops. When executed
with a list of rank arguments (e.g. `python profiler.py 0,1`), it runs different collective communication operations to
profile the bandwidths.

- master.py: the entry script for compiling a model without actually running it.

- worker.py: the entry script for compiling a model and running several training iterations. It needs to be called with
a list of ranks on the machine, e.g., `python worker.py 4,5,6,7` starts 4 instances on the machine, each uses one GPU.
This script also includes the profiling data and need to be edited for each cluster.

- ddp.py: the entry script for running the model with PyTorch DDP. The usage is the same as `worker.py`.

- run_all and run_all_deepspeed: helper scripts for running the same script across machines. They contain ip addresses
in the cluster and need to be edited for each cluster.


## Hardware Dependencies

At least two GPUs are required to run HAP. As HAP is designed for heterogeneous clusters, multiple machines with
different GPU models are needed to fully show HAP’s capabilities. The minimum GPU memory should be at least 12GB. We
recommend a similar setting as used in our experiments (Sec. 7.1), i.e., 2 machines each equipped with 8 NVIDIA V100
GPUs and 6 machines each equipped with 8 NVIDIA P100 GPUs. The inter-machine bandwidth is about 10.4Gbps.


## Software Dependencies

HAP is implemented on PyTorch 1.13.1. All machines should use the same versions of CUDA and NVIDIA drivers that are
compatible with PyTorch 1.13.1. Rust 1.70.0-nightly and Coin CBC 2.9.9 are required to build HAP from source. To
reproduce the results of the baselines, DeepSpeed 0.9.4 is also used.

All software dependencies are included and pre-compiled in the Docker image. However, NVIDIA driver 515.43.04 needs to
be separately installed on the host machines.


## Set-up

This set-up instruction uses the Docker image.

First, ensure that NVIDIA driver 515.43.04 or higher has been installed on the host machines. The installation can be
verified with the `nvidia-smi` command. All machines should use the exact same version of NVIDIA driver. The driver can
be installed by following https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html.

Next, install Docker engine by following https://docs.docker.com/engine/install. After that, download the Docker image
of HAP using `docker pull ylxdzsw/hap:ae`. The image is about 20GB. When finished, start a container instance with
`docker run -d --shm-size="10.24gb" --name hap --gpus all --network host -it ylxdzsw/hap:ae /usr/sbin/sshd`. This will
start an `ssh` instance inside the container on port 3922 for communication between the containers. To access the
container on the host machine, run `docker exec -it hap bash`.

Running HAP involves running the same script on all machines in the cluster. To automate this process, we provide a
helper script `/root/hap/run_all`. Running this script on one of the machines starts the same script on all machines. To
use it, first choose a machine as the controller and run ssh from the controller to all machines (including the
controller itself) with `ssh -p 3922 root@[ip]` and save the ssh fingerprints. Ensure that the controller can access all
workers without further interactions such as confirming fingerprints or typing passwords. Then, edit `/root/hap/run_all`
and replace the ip addresses with the actual ip addresses of the machines.

Finally, check the set-up by running `./run_all worker.py 1` on the controller. It should run 100 iterations of training
and reports the average per-iteration time.


## Experiment E1: [Heterogeneous Cluster] [30 human-minutes + 4 compute-hours]

Train the benchmark models on a heterogeneous cluster and compare the per-iteration training time of HAP and the
baseline systems.

[Preparation]
Assuming that HAP has been set up on a heterogeneous cluster following Sec. A.3, this experiments involves modifying
config.py and running HAP and the baselines.

First, we need to collect profiling data. The device flops can be profiled by running `python profiler.py`. Execute this
command for each type of GPU and replace `device_flops` in `worker.py` with the actual profiling data. `device_flops` is
an array of the flops for all devices. For example, when using 2 V100 GPUs and 6 P100 GPUs, it should be set to an array
of 8 elements, with the first two elements being the profiled flops of the V100 GPU and the last 6 elements being the
profiled flops of the P100 GPU. The collective communication can be profiled by running `./run_all profiler.py 8`, which
automatically runs different collective operators across all machines using 8 GPUs (the second argument to the script)
on each machine. Fill the profiling data in `worker.py`.

Next, modify `config.py` and run `./run_all worker.py k` to obtain the per-iteration training time of HAP, where k is
the number of GPUs to use on each machine. In `config.py`, `model_name` is the benchmark model, where "Vvgg",
"Vtransformer", "Rtransformer" and "Rmoe" correspond to the VGG19, ViT, BERT-Base, and BERT-MoE models. `world_size` is
the total number of GPUs. `master_addr` should be set to the ip address of one of the machines. `cards_per_node` is only
used by the DeepSpeed baseline and should be set to the number of GPUs to use on each machine (same as k). Other fields
should be kept unchanged to reproduce the results reported in the paper.

To run the DP-EV baseline, change the `unscaled_sharding_lengths` in ddp.py to an array of 1 (simulating the same device
flops on each device regardless of their actual types) and run `./run_all ddp.py k` similar to running HAP. To run the
DP-CP baseline, fill `unscaled_sharding_lengths` with the actual profiled flops of each GPU type in the same way as
`device_flops` in `worker.py`.

To run the DeepSpeed baseline, use `./run_all_deepspeed` instead of `./run_all`.

[Execution]
To collect the data for Fig. 12, vary k and the related fields in `config.py` (`model_name`, `world_size`, and
`cards_per_node`), then run HAP and the baselines for each configuration.

[Results]
The experiment scripts print the average per-iteration time and the standard deviation on screen. As the standard
deviation is relatively small in our experiments, we only report the average per-iteration time in Fig. 12. The
experiment script also records the timeline in `trace.json`, which can be load into Chrome Trace Profiling Tool for
further inspection. The results should confirm the claim C1.


## Experiment E2: [Homogeneous Cluster] [30 human-minutes + 4 compute-hours]

Train the benchmark models on a homogeneous cluster and compare the per-iteration training time of HAP and the baseline
systems.

[Preparation]
The preparation is same as in E1, except for that we now use a homogeneous cluster.

[Execution]
Same as in E1.

[Results]
Same as in E1. The results should confirm the claim C2.


## Experiment E3: [Overhead] [5 human-minutes + 5 compute-minutes]

Evaluate the time required by HAP to generate a distributed program.

[Preparation]
This experiment requires only one machine and can run without GPUs. Set `model_name` in `config.py` to "Vtransformer"
for the ViT model and vary the `nlayers` field to experiment with models of different number of layers.

[Execution]
Run python master.py. This script compiles the model without actually running it.

[Results]
The compile time is printed on the screen. The results should confirm the claim C3.
