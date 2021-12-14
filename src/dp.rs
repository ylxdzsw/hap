use std::{rc::Rc, collections::{BTreeMap, BTreeSet}};

use crate::{graph::{self, NodeIndex, TensorIndex, Form, Graph, Node, Signature, SignatureIndex, Tensor}, SVec};

#[derive(Clone)]
pub struct Computation {
    node: NodeIndex,
    signature: graph::SignatureIndex
}

#[derive(Clone)]
pub struct Communication {
    tensor: TensorIndex,
    old_form: graph::Form,
    new_form: graph::Form
}

#[derive(Clone, Default)]
pub struct Stage {
    computations: Vec<Computation>,
    communications: Vec<Communication>,

    acc_cost: f64, // accumulate cost
    prev: Option<Rc<Stage>>
}

struct DuplicaCut { // the cut on a single duplica
    next: NodeIndex,
    state_tensors: Vec<TensorIndex>, // sorted
    state_tensors_reverse_map: BTreeMap<TensorIndex, usize>, // map a tensor to its index in the state_tensors
} // TODO: build all cuts on beginning and refers to it

impl DuplicaCut {
    fn new(next: NodeIndex, state_tensors: Vec<TensorIndex>) -> DuplicaCut {
        let reverse_map = state_tensors.iter().enumerate().map(|(i, &tensor_index)| (tensor_index, i)).collect();
        DuplicaCut { next, state_tensors, state_tensors_reverse_map: reverse_map }
    }

    fn get_all_in_graph(graph: &Graph) -> Vec<DuplicaCut> {
        let mut state_tensors = vec![vec![]; graph.nodes.len()];

        for (tensor_id, tensor) in graph.tensors.iter().enumerate() {
            let start = tensor.producer.0;
            let end = tensor.consumers.iter().map(|x| x.0).max().unwrap();
            #[allow(clippy::needless_range_loop)]
            for i in start+1..end+1 {
                state_tensors[i].push(TensorIndex(tensor_id))
            }
        }

        state_tensors.into_iter().enumerate().map(|(i, x)| {
            DuplicaCut::new(NodeIndex(i), x)
        }).collect()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct CutIndex(pub usize);

#[derive(Clone)]
pub struct DuplicaState {
    cut: CutIndex,
    next_communicatable: TensorIndex, // heuristic: the smallest index tensor that can communicate
    available_forms: Vec<BTreeSet<Form>>, // all available forms for each state tensor
}

impl DuplicaState {
    fn advance_computation(&self, dp2: &DP2) -> Vec<(DuplicaState, Computation)> {
        if dp2.nodes.len() <= dp2.get_cut(self.cut).next.0 {
            return vec![]
        }

        let next_node = dp2.get_node(dp2.get_cut(self.cut).next);
        let mut result = vec![];

        for (signature_id, signature) in next_node.signatures.iter().enumerate() {
            if !self.is_signature_compatable(dp2, next_node, signature) {
                continue
            }

            let next_cut = &dp2.cuts[self.cut.0+1];
            let mut next_state = DuplicaState {
                cut: CutIndex(self.cut.0+1),
                next_communicatable: self.next_communicatable,
                available_forms: next_cut.state_tensors.iter().map(|tensor_index| {
                    dp2.get_cut(self.cut).state_tensors_reverse_map.get(tensor_index).map(|&i| self.available_forms[i].clone()).unwrap_or_default()
                }).collect()
            };
            for (output_tensor_index, &output_tensor_form) in next_node.outputs.iter().zip(signature.output_forms.iter()) {
                let i = next_cut.state_tensors_reverse_map[output_tensor_index];
                next_state.available_forms[i].insert(output_tensor_form);
            }
            let computation = Computation { node: dp2.get_cut(self.cut).next, signature: SignatureIndex(signature_id) };
            result.push((next_state, computation))
        }

        result
    }

    fn advance_communication(&self, dp2: &DP2) -> Vec<(DuplicaState, Communication)> {
        let mut result = vec![];

        for (i, (&tensor_index, available_forms)) in dp2.get_cut(self.cut).state_tensors.iter().zip(self.available_forms.iter()).enumerate() {
            if tensor_index < self.next_communicatable {
                continue
            }

            for &form in dp2.get_tensor(tensor_index).consumer_forms.iter() {
                if available_forms.contains(&form) {
                    continue
                }

                // Note: we could keep next_communicatable to be the same, so each tensor may communicate multiple times. Also we can select one old_form that performs best.
                let mut next_state = DuplicaState {
                    cut: self.cut,
                    next_communicatable: TensorIndex(tensor_index.0 + 1),
                    available_forms: self.available_forms.clone()
                };
                let communication = Communication {
                    tensor: tensor_index,
                    old_form: *available_forms.iter().next().unwrap(),
                    new_form: form
                };
                next_state.available_forms[i].insert(form);

                result.push((next_state, communication))
            }
        }

        result
    }

    fn is_signature_compatable(&self, dp2: &DP2, node: &Node, signature: &Signature) -> bool {
        for (input_tensor_index, input_tensor_form) in node.inputs.iter().zip(signature.input_forms.iter()) {
            let available_forms = self.available_forms[dp2.get_cut(self.cut).state_tensors_reverse_map[input_tensor_index]].clone();
            if available_forms.contains(input_tensor_form) {
                continue
            }
            return false
        }

        true
    }
}

struct State(DuplicaState, DuplicaState, Rc<Stage>);

impl State {
    fn empty() -> Self {
        let a = DuplicaState { cut: CutIndex(0), next_communicatable: TensorIndex(0), available_forms: vec![] };
        let b = DuplicaState { cut: CutIndex(0), next_communicatable: TensorIndex(0), available_forms: vec![] };
        let empty_stage = Rc::new(Stage::default());
        State(a, b, empty_stage)
    }
}

struct DP2<'g> {
    nodes: &'g [Node],
    tensors: &'g [Tensor],
    n: usize,
    cuts: Vec<DuplicaCut>,
    pareto: Vec<Vec<Vec<State>>>

}

impl<'g> DP2<'g> {
    fn new(graph: &Graph) -> DP2 {
        let Graph { nodes, tensors } = graph;
        let n = nodes.len();

        let mut cuts = DuplicaCut::get_all_in_graph(graph);
        let mut pareto: Vec<Vec<Vec<State>>> = // the (i, j)-th element is a list of states and corresponding best stage when the first duplica is on cut i and the second is on cut j
            (0..n).map(|_| (0..n).map(|_| vec![]).collect()).collect();

        pareto[0][0].push(State::empty());
        DP2 { nodes, tensors, n, cuts, pareto }
    }

    fn get_node(&self, node_index: NodeIndex) -> &Node {
        &self.nodes[node_index.0]
    }

    fn get_tensor(&self, tensor_index: TensorIndex) -> &Tensor {
        &self.tensors[tensor_index.0]
    }

    fn get_cut(&self, cut_index: CutIndex) -> &DuplicaCut {
        &self.cuts[cut_index.0]
    }

    fn explore_next_stage(&mut self, state: State, stage: Stage) {
        let State(a, b, prev) = state;

        // 1. computation on a, possibly communication on b
        let mut next_duplica_state_a = a.clone();
        for n_comp in (0..5) {
            if cut_index_a + n_comp - cut_index_b > 5 { // heuristic
                break
            }

        }
    }

    fn dp(&mut self) {
        #[allow(clippy::needless_range_loop)]
        for cut_index_a in 0..self.n {
            for cut_index_b in 0..self.n {
                if cut_index_a < cut_index_b || cut_index_a - cut_index_b > 5 { // heuristic
                    continue
                }

                for state in self.pareto[cut_index_a][cut_index_b].iter() {


                }
            }
        }
    }
}

// dp for duplexed graph
pub fn dp2(graph: &Graph) {
    let mut dp2 = DP2::new(graph);


}
