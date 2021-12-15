use std::{rc::Rc, collections::{BTreeMap, BTreeSet}, ops::Index, cell::RefCell};

use crate::{graph::{self, NodeIndex, TensorIndex, Form, Graph, Node, Signature, SignatureIndex, Tensor}, SVec};

type Cache<T> = Rc<RefCell<Option<T>>>;

const PROGRESS_LIMIT: usize = 5;

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
    computation_on_first_duplica: bool,

    computations: Vec<Computation>,
    communications: Vec<Communication>,

    acc_cost: f64, // accumulate cost
    prev: Option<Rc<Stage>>
}

struct DuplicaCut { // the cut on a single duplica
    next: NodeIndex,
    state_tensors: Vec<TensorIndex>, // sorted
    state_tensors_reverse_map: BTreeMap<TensorIndex, usize>, // map a tensor to its index in the state_tensors
}

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

    next_computations: Cache<Vec<(DuplicaState, Computation)>>,
    next_communications: Cache<Vec<(DuplicaState, Communication)>>,

    possible_computations: Cache<Vec<(DuplicaState, Vec<Computation>)>>,
    possible_communications: Cache<Vec<(DuplicaState, Vec<Communication>)>>,
}

impl DuplicaState {
    fn new(cut: CutIndex, next_communicatable: TensorIndex, available_forms: Vec<BTreeSet<Form>>) -> DuplicaState {
        DuplicaState {
            cut, next_communicatable, available_forms,
            next_computations: Rc::new(RefCell::new(None)),
            next_communications: Rc::new(RefCell::new(None)),
            possible_computations: Rc::new(RefCell::new(None)),
            possible_communications: Rc::new(RefCell::new(None))
        }
    }

    fn get_next_computations(&self, g: &GraphInfo) -> Vec<(DuplicaState, Computation)> {
        self.next_computations.borrow_mut().get_or_insert_with(|| {
            if g.len() <= g[self.cut].next.0 {
                return vec![]
            }

            let next_node = &g[g[self.cut].next];
            let mut result = vec![];

            for (signature_id, signature) in next_node.signatures.iter().enumerate() {
                if !self.is_signature_compatable(g, next_node, signature) {
                    continue
                }

                let next_cut = &g[CutIndex(self.cut.0+1)];
                let mut next_state = DuplicaState::new(
                    CutIndex(self.cut.0+1),
                    self.next_communicatable,
                    next_cut.state_tensors.iter().map(|tensor_index| {
                        g[self.cut].state_tensors_reverse_map.get(tensor_index).map(|&i| self.available_forms[i].clone()).unwrap_or_default()
                    }).collect()
                );
                for (output_tensor_index, &output_tensor_form) in next_node.outputs.iter().zip(signature.output_forms.iter()) {
                    let i = next_cut.state_tensors_reverse_map[output_tensor_index];
                    next_state.available_forms[i].insert(output_tensor_form);
                }
                let computation = Computation { node: g[self.cut].next, signature: SignatureIndex(signature_id) };
                result.push((next_state, computation))
            }

            result
        }).clone()
    }

    fn get_possible_computations(&self, g: &GraphInfo) -> Vec<(DuplicaState, Vec<Computation>)> {
        self.possible_computations.borrow_mut().get_or_insert_with(|| {
            let mut results: Vec<_> = self.get_next_computations(g).into_iter().map(|(state, computation)| (state, vec![computation])).collect();
            let mut next_level = results.clone();
            while !next_level.is_empty() {
                let mut next_next_level = vec![];
                for (state, computations) in next_level {
                    if computations.len() >= PROGRESS_LIMIT {
                        continue
                    }
                    for (next_state, next_computation) in state.get_next_computations(g) {
                        let mut next_computations = computations.clone();
                        next_computations.push(next_computation);
                        next_next_level.push((next_state, next_computations))
                    }
                }
                results.extend(next_next_level.iter().cloned());
                next_level = next_next_level;
            }
            results
        }).clone()
    }

    fn get_next_communications(&self, g: &GraphInfo) -> Vec<(DuplicaState, Communication)> {
        let mut result = vec![];

        for (i, (&tensor_index, available_forms)) in g[self.cut].state_tensors.iter().zip(self.available_forms.iter()).enumerate() {
            if tensor_index < self.next_communicatable {
                continue
            }

            for &form in g[tensor_index].consumer_forms.iter() {
                if available_forms.contains(&form) {
                    continue
                }

                // Note: we could keep next_communicatable to be the same, so each tensor may communicate multiple times. Also we can select one old_form that performs best.
                let mut next_state = DuplicaState::new(
                    self.cut,
                    TensorIndex(tensor_index.0 + 1),
                    self.available_forms.clone()
                );
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

    fn get_possible_communications(&self, g: &GraphInfo) -> Vec<(DuplicaState, Vec<Communication>)> {
        // this may be optimized
        self.possible_communications.borrow_mut().get_or_insert_with(|| {
            let mut results: Vec<_> = self.get_next_communications(g).into_iter().map(|(state, communication)| (state, vec![communication])).collect();
            let mut next_level = results.clone();
            while !next_level.is_empty() {
                let mut next_next_level = vec![];
                for (state, communications) in next_level {
                    for (next_state, next_communication) in state.get_next_communications(g) {
                        let mut next_communications = communications.clone();
                        next_communications.push(next_communication);
                        next_next_level.push((next_state, next_communications))
                    }
                }
                results.extend(next_next_level.iter().cloned());
                next_level = next_next_level;
            }
            results
        }).clone()
    }

    fn is_signature_compatable(&self, gh: &GraphInfo, node: &Node, signature: &Signature) -> bool {
        for (input_tensor_index, input_tensor_form) in node.inputs.iter().zip(signature.input_forms.iter()) {
            let available_forms = self.available_forms[gh[self.cut].state_tensors_reverse_map[input_tensor_index]].clone();
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
    fn empty(gh: &GraphInfo) -> Self {
        let a = DuplicaState::new(CutIndex(0), TensorIndex(0), vec![]);
        let b = DuplicaState::new(CutIndex(0), TensorIndex(0), vec![]);
        let empty_stage = Rc::new(Stage::default());
        State(a, b, empty_stage)
    }
}

struct GraphInfo<'g> {
    nodes: &'g [Node],
    tensors: &'g [Tensor],
    cuts: Vec<DuplicaCut>,
}

impl<'g> Index<NodeIndex> for GraphInfo<'g> {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl<'g> Index<TensorIndex> for GraphInfo<'g> {
    type Output = Tensor;

    fn index(&self, index: TensorIndex) -> &Self::Output {
        &self.tensors[index.0]
    }
}

impl<'g> Index<CutIndex> for GraphInfo<'g> {
    type Output = DuplicaCut;

    fn index(&self, index: CutIndex) -> &Self::Output {
        &self.cuts[index.0]
    }
}

impl<'g> GraphInfo<'g> {
    fn new(graph: &Graph) -> GraphInfo {
        let Graph { nodes, tensors } = graph;
        let cuts = DuplicaCut::get_all_in_graph(graph);
        GraphInfo { nodes, tensors, cuts }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

// dp for duplexed graph
pub fn dp2(graph: &Graph) {
    let g = GraphInfo::new(graph);
    let n = g.len();

    let mut pareto: Vec<Vec<Vec<State>>> = // the (i, j)-th element is a list of states and corresponding best stage when the first duplica is on cut i and the second is on cut j
        (0..n).map(|_| (0..n).map(|_| vec![]).collect()).collect();

    let initial_state = State::empty(&g);
    pareto[0][0].push(initial_state);

    #[allow(clippy::needless_range_loop)]
    for cut_index_a in 0..n {
        for cut_index_b in cut_index_a.saturating_sub(PROGRESS_LIMIT)..cut_index_a {
            for state in pareto[cut_index_a][cut_index_b].iter() {


            }
        }
    }
}

fn explore_next_stage(g: &GraphInfo, pareto: &mut Vec<Vec<Vec<State>>>, state: State) {
    let State(a, b, prev_stage) = state;

    // TODO: check the progress of the two batches

    // 1. computation on a, possibly communication on b
    for (state_a, computations) in a.get_possible_computations(g) {
        for (state_b, communications) in b.get_possible_communications(g) {
            let stage = Rc::new(Stage {
                computation_on_first_duplica: true,
                computations: computations.clone(),
                communications,
                acc_cost: prev_stage.acc_cost,
                prev: Some(prev_stage.clone()),
            });
            let new_state = State(state_a.clone(), state_b, stage.clone());
            append_path(g, pareto, new_state, stage);
        }

        let stage = Rc::new(Stage {
            computation_on_first_duplica: true,
            computations,
            communications: vec![],
            acc_cost: prev_stage.acc_cost,
            prev: Some(prev_stage.clone()),
        });
        let new_state = State(state_a, b.clone(), stage.clone());
        append_path(g, pareto, new_state, stage);
    }

    // 2. computation on b, possibly communication on a
    for (state_b, computations) in b.get_possible_computations(g) {
        for (state_a, communications) in a.get_possible_communications(g) {
            let stage = Rc::new(Stage {
                computation_on_first_duplica: false,
                computations: computations.clone(),
                communications,
                acc_cost: prev_stage.acc_cost,
                prev: Some(prev_stage.clone()),
            });
            let new_state = State(state_a, state_b.clone(), stage.clone());
            append_path(g, pareto, new_state, stage);
        }

        let stage = Rc::new(Stage {
            computation_on_first_duplica: false,
            computations,
            communications: vec![],
            acc_cost: prev_stage.acc_cost,
            prev: Some(prev_stage.clone()),
        });
        let new_state = State(a.clone(), state_b, stage.clone());
        append_path(g, pareto, new_state, stage);
    }

    // 3. todo: only communication?
}

// try append a stage that leads to state in the pareto
fn append_path(g: &GraphInfo, pareto: &mut Vec<Vec<Vec<State>>>, state: State, stage: Rc<Stage>) {

}
