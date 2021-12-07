use std::rc::Rc;

use crate::{graph::{self, NodeIndex, TensorIndex, Form, Graph}, SVec};

pub struct Computation {
    node: NodeIndex,
    signature: graph::SignatureIndex
}

pub struct Communication {
    tensor: TensorIndex,
    old_form: graph::Form,
    new_form: graph::Form
}

pub struct Stage {
    computations: Vec<Computation>,
    communications: Vec<Communication>,

    acc_cost: f64, // accumulate cost
    prev: Rc<Stage>
}

pub struct DuplicaState {
    cut: DuplicaCut,
    communication_dep: TensorIndex, // if enabled, future communications can only happen on tensors that has larger index
    available_forms: Vec<SVec<Form>>, // all available forms for each state tensor
}

struct DuplicaCut { // the cut on a single duplica
    next: NodeIndex,
    state_tensors: Vec<TensorIndex>, // tensors on the cut, sorted by tensor index
} // TODO: build all cuts on beginning and refers to it

impl DuplicaCut {
    fn get_all_in_graph(graph: &Graph) -> Vec<DuplicaCut> {
        todo!()
    }
}

// dp for duplexed graph
pub fn dp2(graph: &Graph) {
    struct State(DuplicaState, DuplicaState, Rc<Stage>);

    let Graph { nodes, tensors } = graph;
    let n = nodes.len();

    let mut cuts = DuplicaCut::get_all_in_graph(graph);
    let mut pareto: Vec<Vec<Vec<(DuplicaState, DuplicaState, Option<Rc<Stage>>)>>> = todo!(); // the (i, j)-th element is a list of states and corresponding best stage when the first duplica is on cut i and the second is on cut j

    let mut front: Vec<()> = vec![Default::default()]; // the states to consider next, sorted
    let mut pareto: Vec<Vec<Rc<Stage>>> = vec![];


}
