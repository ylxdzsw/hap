crate::new_usize_type!(pub, DeviceId);


enum PartitionDimension {
    Value,
    Dim(usize)
}

enum Participant {
    Device(DeviceId),
    Group(Vec<DeviceId>, PartitionDimension)
}

struct Level {
    members: Vec<Participant>,
    aggregator: Participant
}

// treat machine as the basic heterogenous unit. Encapsule the cards in a machine (each machine get a portion of task. It may use any parallelism to fulfill it)

// no need to explicit hierachical communication mechanism. Just need a recursive sharding program that can handle graph with communication operations.

// 1. draw a graph showing homogenous cluster (TPU, DGX), hierarchical cluster, and our topology.
// 2. two level of communication. First level is inter-cluster and use our topology-aware routing (it is CPU-only). The second level is just NCCL.

// or make a small comparizon with direct NCCL or hierarchical comm.
