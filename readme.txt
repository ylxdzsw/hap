Source code for "Accelerating Large-Scale Distributed Neural Network Training with SPMD Parallelism"

We recommend our Docker image (https://hub.docker.com/r/ylxdzsw/spmd) for reproducing the experiments in the paper.

=== Experiment VI.B ===

Experiment Summary: We evaluate HiDup, DeepSpeed and FastMoE in a cluster with up to 64 GPUs.

Software:
    - HiDup (provided in artifacts)
    - DeepSpeed 0.5.10 with patches
    - FastMoE 0.3.0
    - PyTorch 1.10.0
    - Python 3.7.11
    - CUDA 11.4.1
    - NVIDIA GPU driver 470.82.01
    - CUDNN 8.2.4

We patch DeepSpeed by editing line 282 of "moe/sharded_moe.py" and replacing "min_capacity" with "1". This variable is not used but causes undefined variable error without this patch. The docker image provided in artifacts is already patched.

Hardware: We use 8 ecs.gn6vc8g1.16xlarge instances in Alibaba Cloud (https://aliyun.com).
    - CPU: 64vCPU Intel Xeon(Skylake) Platinum 8163
    - RAM: 256GiB
    - GPU: 8 NVIDIA V100 GPUs
    - Bandwidth: 9.71Gbps (measured using iperf3)

Workflow: The commands are executed in the "exp" directory of HiDup.
    1. Edit "master_addr" and "master_port" in "config.py" according to the cluster configuration.
    2. Run "python profiler.py" on any machine to get the estimated device flops. Set "profiler_data.device_flops" in "config.py" to this value.
    3. Set "world_size" in "config.py" to the total number of GPUs.
    4. Run "NODERANK=a CPN=b python profiler.py" on each machine to measure the bandwidth of different collective operations. "a" is the rank of the machine (starts from 0) and "b" is the number of GPUs to use on each machine. Edit each field of "profiler_data" in "config.py" accordingly.
    5. Set "model_name" in "config.py" to any of the following: "Rmoe", "Rswitch", "Vmoe" and "Vswitch". They correspond to "BERT-SGMoE", "BERT-Switch", "ViT-SGMoE" and "ViT-Switch" models, respectively.
    6. Run "python strategy.py" on any machine. It will print the searched strategy and save it to a file with the name "strategy_[model name]".
    7. Copy the edited "config.py" and the generated strategy file to all machines.
    8. Run "NODERANK=a CPN=b python train.py" on each machine to get the average per-iteration training time for HiDup.
    9. Run "NODERANK=a CPN=b python fastmoe/train.py" on each machine to get the average per-iteration training time for FastMoE.
    10. Run "NODERANK=a CPN=b bash deepspeed/run.sh" on each machine to get the average per-iteration training time for DeepSpeed.
    11. Repeat steps 3 to 10 and vary the model and number of GPUs to collect data for Figure 7.


=== Experiment VI.C ===

Experiment Summary: We evaluate HiDup, DeepSpeed, FastMoE, Horovod and PyTorch DDP in a single machine of 8 GPUs.

Software:
    - Horovod 0.24.2
    - Others are the same as in Experiment VI.B

Hardware: Same as in Experiment VI.B.

Workflow:
    1. Follow steps 1 to 10 of Experiment VI.B.
    2. Run "NODERANK=0 CPN=b python ddp.py" on each machine to get the per-iteration training time for PyTorch DDP. "b" is the number of GPUs to use on each machine.
    3. Run "NODERANK=0 CPN=b python hvd.py" on each machine to get the per-iteration training time for Horovod.
    4. Repeat steps 1 to 3 and vary the model and number of GPUs to collect data for Figure 8.


=== Experiment VI.D ===

Experiment Summary: We evaluate HiDup, DeepSpeed and FastMoE on 2 machines under different bandwidth levels.

Software:
    - iproute2 4.15.0-2ubuntu1.3
    - Others are the same as in Experiment VI.B

Hardware:.
    - CPU: Intel(R) Xeon(R) Gold 6230
    - RAM: 251G
    - GPU: 4 NVIDIA V100 GPUs
    - Bandwidth: 37.5Gbps (measured using iperf3)

Workflow:
    1. Run "tc qdisc add dev eth0 root tbf rate 30gibit latency 60ms burst 60m" on each machine to limit the available bandwidth. "eth0" should be replaced with the actual network interface and "30gibit" can be changed for the experiment.
    2. Follow steps 1 to 11 of Experiment VI.B to collect the results under the bandwidth level specified in step 1.
    3. Run "tc qdisc del dev eth0 root" on each machine to reset the bandwidth limit.
    4. Repeat steps 1 to 3 and change the bandwidth limit to collect data for Figure 9.


=== Experiment VI.E ===

Experiment Summary: We evaluate the impact of our computation and communication time estimation on the strategy found by HiDup.

Software: Same as in Experiment VI.D

Hardware: Same as in Experiment VI.D

Workflow:
    1. Change "profile_noise" in "config.py" to the noise level. For example, "profile_noise = 0.8" corresponds to "Noise level 80%" in Table I.
    2. Follow steps 1 to 8 of Experiment VI.B to collect the results for the strategy based on the noisy estimated times. Repeat 10 times under the same noise level and calculate the average relative time.
    3. Repeat steps 1 and 2 with different noise levels to collect data for Table I.


=== Experiment VI.F ===

Experiment Summary: We study HiDup's strategy search time regarding the number of GPUs and the number of layers for the BERT-SGMoE model.

Software: Same as in Experiment VI.D

Hardware: Same as in Experiment VI.D

Workflow:
    1. Change "nlayers" and "world_size" in "config.py", which are the number of layers and the number of GPUs, respectively.
    2. Run "time python strategy.py" to record the strategy search time.
    3. Repeat steps 1 and 2 with different numbers of layers and numbers of GPUs to collect data for Figure 10.
