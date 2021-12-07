use std::rc::Rc;

use crate::graph;

#[derive(Clone, Copy, Debug)]
pub struct Node {
    raw_node: graph::NodeIndex,
    duplica_id: usize,

}

#[derive(Clone, Copy, Debug)]
pub struct Tensor {
    raw_tensor: graph::TensorIndex,
    duplica_id: usize,

}


pub struct Computation {
    node: Node,
    signature: graph::SignatureIndex
}

pub struct Communication {
    tensor: Tensor,
    old_form: graph::Form,
    new_form: graph::Form
}

pub struct Stage {
    computations: Vec<Computation>,
    communications: Vec<Communication>,

    prev: Rc<Stage>
}

