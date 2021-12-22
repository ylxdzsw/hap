use std::{rc::Rc, collections::{BTreeMap, BTreeSet}, ops::Index, cell::RefCell};

use float_ord::FloatOrd;
use oh_my_rust::*;

use crate::{graph::{NodeIndex, TensorIndex, Form, Node, Signature, SignatureIndex, Tensor}, SVec, smallvec, CTRLC_RECEIVED};

type Cache<T> = Rc<RefCell<Option<T>>>;

const PROGRESS_LIMIT: usize = 5;

#[derive(Clone)]
pub struct Computation {
    node: NodeIndex,
    signature: SignatureIndex
}

impl Computation {
    fn cost(&self, g: Graph) -> f64 {
        g[self.node].signatures[self.signature.0].cost
    }
}

#[derive(Clone)]
pub struct Communication {
    tensor: TensorIndex,
    old_form: Form,
    new_form: Form
}

impl Communication {
    fn cost(&self, g: Graph) -> f64 {
        // g[self.tensor].size
        todo!()
    }
}

/// invariant: comp_a and comp_b cannot both be non-empty. The same applies for comm_a and comm_b
#[derive(Clone, Default)]
pub struct Stage {
    computations_a: SVec<Computation, PROGRESS_LIMIT>,
    computations_b: SVec<Computation, PROGRESS_LIMIT>,

    communications_a: SVec<Communication>,
    communications_b: SVec<Communication>,

    acc_cost: f64, // accumulate cost = prev.acc_cost + max(self.comp, self.comm)
    prev: Option<Rc<Stage>>
}

impl Stage {
    // check if a stage is overlapping (has both computation and communication). Non-overlapping statges can only be followed by an overlapping statge.
    // empty statges (has neither computation and communication) is treated as overlapping to allow the first stage to be non-overlapping
    fn is_overlapping(&self) -> bool {
        !(self.has_computation() ^ self.has_communication())
    }

    fn has_computation(&self) -> bool {
        !self.computations_a.is_empty() || !self.computations_b.is_empty()
    }

    fn has_communication(&self) -> bool {
        !self.communications_a.is_empty() || !self.communications_b.is_empty()
    }
}

struct DuplicaCut { // the cut on a single duplica
    next_node: NodeIndex,
    state_tensors: Vec<TensorIndex>, // sorted
    state_tensors_reverse_map: BTreeMap<TensorIndex, usize>, // map a tensor to its index in the state_tensors
}

impl DuplicaCut {
    fn new(next_node: NodeIndex, state_tensors: Vec<TensorIndex>) -> DuplicaCut {
        let reverse_map = state_tensors.iter().enumerate().map(|(i, &tensor_index)| (tensor_index, i)).collect();
        DuplicaCut { next_node, state_tensors, state_tensors_reverse_map: reverse_map }
    }

    fn get_all_in_graph(g: &Graph) -> Vec<DuplicaCut> {
        let mut state_tensors = vec![vec![]; g.nodes.len()];

        for (tensor_id, tensor) in g.tensors.iter().enumerate() {
            if tensor.consumers.is_empty() { // for some .shape tensors, only some of the outputs (size of specific dimensions) are used
                continue
            }
            let start = tensor.producer.0;
            let end = tensor.consumers.iter().map(|x| x.0).max().unwrap();
            #[allow(clippy::needless_range_loop)]
            for i in start+1..end+1 {
                state_tensors[i].push(TensorIndex(tensor_id))
            }
        }

        state_tensors.push(vec![]); // a trailing cut that cuts after the last node

        for (i, state_tensor) in state_tensors.iter().enumerate() {
            info!("cut {} has {} tensors, totaling {} states",
                i, state_tensor.len(), state_tensor.iter().map(|&x| g[x].consumer_forms.len()).product::<usize>()
            )
        }

        state_tensors.into_iter().enumerate().map(|(i, x)| {
            DuplicaCut::new(NodeIndex(i), x)
        }).collect()
    }
}

crate::new_index_type!(pub, CutIndex);

// a collection that has one or two forms, ensuring sorted in the case of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OneOrTwoForms { One(Form), Two(Form, Form) }

impl OneOrTwoForms {
    fn one(form: Form) -> OneOrTwoForms {
        OneOrTwoForms::One(form)
    }

    fn two(a: Form, b: Form) -> OneOrTwoForms {
        OneOrTwoForms::Two(std::cmp::min(a, b), std::cmp::max(a, b))
    }

    fn insert(&mut self, form: Form) {
        *self = OneOrTwoForms::two(self.unwrap(), form)
    }

    fn contains(&self, form: Form) -> bool {
        match *self {
            OneOrTwoForms::One(a) => a == form,
            OneOrTwoForms::Two(a, b) => a == form || b == form,
        }
    }

    fn len(&self) -> usize {
        match self {
            OneOrTwoForms::One(..) => 1,
            OneOrTwoForms::Two(..) => 2,
        }
    }

    fn unwrap(self) -> Form {
        match self {
            OneOrTwoForms::One(form) => form,
            OneOrTwoForms::Two(..) => panic!("unwrap two forms")
        }
    }
}

#[derive(Clone)]
pub struct DuplicaState {
    cut: CutIndex,
    next_communicatable: TensorIndex, // heuristic: the smallest index tensor that can communicate
    available_forms: Vec<OneOrTwoForms>, // all available forms for each state tensor

    next_computations: Cache<SVec<(DuplicaState, Computation)>>,
    next_communications: Cache<SVec<(DuplicaState, Communication)>>,

    possible_computations: Cache<Vec<(DuplicaState, SVec<Computation, PROGRESS_LIMIT>)>>,
    possible_communications: Cache<Vec<(DuplicaState, SVec<Communication>)>>,
}

impl DuplicaState {
    fn new(cut: CutIndex, next_communicatable: TensorIndex, available_forms: Vec<OneOrTwoForms>) -> DuplicaState {
        DuplicaState {
            cut, next_communicatable, available_forms,
            next_computations: Default::default(),
            next_communications: Default::default(),
            possible_computations: Default::default(),
            possible_communications: Default::default()
        }
    }

    fn get_next_computations(&self, g: &Graph) -> SVec<(DuplicaState, Computation)> {
        self.next_computations.borrow_mut().get_or_insert_with(|| {
            if g.n_nodes() <= g[self.cut].next_node.0 {
                return smallvec![]
            }

            let next_node = &g[g[self.cut].next_node];
            let mut result = smallvec![];

            for (signature_id, signature) in next_node.signatures.iter().enumerate() {
                if !self.is_signature_compatable(g, next_node, signature) {
                    continue
                }

                let next_cut = &g[self.cut+1];
                let mut next_state = DuplicaState::new(
                    self.cut + 1,
                    self.next_communicatable,
                    next_cut.state_tensors.iter().map(|tensor_index| {
                        g[self.cut].state_tensors_reverse_map.get(tensor_index)
                            .map(|&i| self.available_forms[i])
                            .unwrap_or_else(|| {
                                let i = next_node.outputs.iter().position(|x| x == tensor_index).unwrap();
                                OneOrTwoForms::one(signature.output_forms[i])
                            })
                    }).collect()
                );
                let computation = Computation { node: g[self.cut].next_node, signature: SignatureIndex(signature_id) };
                result.push((next_state, computation))
            }

            result
        }).clone()
    }

    fn get_possible_computations(&self, g: &Graph) -> Vec<(DuplicaState, SVec<Computation, PROGRESS_LIMIT>)> {
        self.possible_computations.borrow_mut().get_or_insert_with(|| {
            let mut results: Vec<_> = self.get_next_computations(g).into_iter().map(|(state, computation)| (state, smallvec![computation])).collect();

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

    fn get_next_communications(&self, g: &Graph) -> SVec<(DuplicaState, Communication)> {
        let mut result = smallvec![];

        for (i, (&tensor_index, available_forms)) in g[self.cut].state_tensors.iter().zip(self.available_forms.iter()).enumerate() {
            if tensor_index < self.next_communicatable {
                continue
            }

            if available_forms.len() >= 2 {
                continue
            }

            for &form in g[tensor_index].consumer_forms.iter() {
                if available_forms.contains(form) {
                    continue
                }

                // Note: we could keep next_communicatable to be the same, so each tensor may communicate multiple times. Also we can select one old_form that performs best.
                let mut next_state = DuplicaState::new(
                    self.cut,
                    tensor_index + 1,
                    self.available_forms.clone()
                );
                let communication = Communication {
                    tensor: tensor_index,
                    old_form: available_forms.unwrap(),
                    new_form: form
                };
                next_state.available_forms[i].insert(form);

                result.push((next_state, communication))
            }
        }

        result
    }

    fn get_possible_communications(&self, g: &Graph) -> Vec<(DuplicaState, SVec<Communication>)> {
        // this may be optimized
        self.possible_communications.borrow_mut().get_or_insert_with(|| {
            let mut results: Vec<_> = self.get_next_communications(g).into_iter().map(|(state, communication)| (state, smallvec![communication])).collect();
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

    fn is_signature_compatable(&self, gh: &Graph, node: &Node, signature: &Signature) -> bool {
        for (input_tensor_index, input_tensor_form) in node.inputs.iter().zip(signature.input_forms.iter()) {
            let available_forms = self.available_forms[gh[self.cut].state_tensors_reverse_map[input_tensor_index]];
            if available_forms.contains(*input_tensor_form) {
                continue
            }
            return false
        }

        true
    }

    fn progress(&self) -> usize {
        self.cut.0
    }
}

#[derive(Clone)]
struct State(DuplicaState, DuplicaState, Rc<Stage>);

impl State {
    fn empty(gh: &Graph) -> Self {
        let a = DuplicaState::new(0.into(), 0.into(), vec![]);
        let b = DuplicaState::new(0.into(), 0.into(), vec![]);
        let empty_stage = Rc::new(Stage::default());
        State(a, b, empty_stage)
    }
}

impl State {
    fn get_pareto_cell_key(&self) -> ParetoCellKey {
        let State(a, b, _) = self;
        (a.available_forms.clone(), b.available_forms.clone())
    }
}

struct Graph<'g> {
    nodes: &'g [Node],
    tensors: &'g [Tensor],
    cuts: Vec<DuplicaCut>,
}

impl<'g> Index<NodeIndex> for Graph<'g> {
    type Output = Node;

    fn index(&self, index: NodeIndex) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl<'g> Index<TensorIndex> for Graph<'g> {
    type Output = Tensor;

    fn index(&self, index: TensorIndex) -> &Self::Output {
        &self.tensors[index.0]
    }
}

impl<'g> Index<CutIndex> for Graph<'g> {
    type Output = DuplicaCut;

    fn index(&self, index: CutIndex) -> &Self::Output {
        &self.cuts[index.0]
    }
}

impl<'g> Graph<'g> {
    fn new(graph: &crate::graph::Graph) -> Graph {
        let crate::graph::Graph { nodes, tensors } = graph;
        let mut result = Graph { nodes, tensors, cuts: vec![] };
        result.cuts = DuplicaCut::get_all_in_graph(&result);
        result
    }

    fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn n_cuts(&self) -> usize {
        self.cuts.len()
    }
}

// dp for duplexed graph
pub fn dp2(graph: &crate::graph::Graph) {
    let g = Graph::new(graph);
    let n_cut = g.n_cuts();

    let mut pareto: Pareto = // the (i, j)-th element is a list of states and corresponding best stage when the first duplica is on cut i and the second is on cut j
        (0..n_cut).map(|_| (0..n_cut).map(|_| Default::default()).collect()).collect();

    let initial_state = State::empty(&g);
    pareto[0][0].insert(initial_state.get_pareto_cell_key(), initial_state);

    for cut_index_a in 0..n_cut {
        for cut_index_b in cut_index_a.saturating_sub(PROGRESS_LIMIT)..=cut_index_a {
            info!("expanding on ({},{}) with {} states", cut_index_a, cut_index_b, pareto[cut_index_a][cut_index_b].len());
            let mut visited: BTreeSet<ParetoCellKey> = BTreeSet::default();
            loop {
                let next_pair = pareto[cut_index_a][cut_index_b].iter()
                    .filter(|(k, v)| !visited.contains(k))
                    .min_by_key(|(k, v)| FloatOrd(v.2.acc_cost))
                    .map(|(k, v)| (k.clone(), v.clone()));
                if let Some((k, v)) = next_pair {
                    visited.insert(k);
                    explore_next_stage(&g, &mut pareto, v);
                } else {
                    break
                }
            }
            info!("expanded on ({},{}) with {} states", cut_index_a, cut_index_b, pareto[cut_index_a][cut_index_b].len());
            // eager free memory
            pareto[cut_index_a][cut_index_b].clear()
        }
    }
}

type ParetoCellKey = (Vec<OneOrTwoForms>, Vec<OneOrTwoForms>);
type Pareto = Vec<Vec<BTreeMap<ParetoCellKey, State>>>;

#[allow(clippy::redundant_clone)]
fn explore_next_stage(g: &Graph, pareto: &mut Pareto, state: State) {
    let State(a, b, prev_stage) = state;

    // 1. computation on a, communication on b
    for (new_a, computations) in a.get_possible_computations(g) {
        if b.progress() + PROGRESS_LIMIT < new_a.progress() { // exceed progress limit
            continue
        }
        for (new_b, communications) in b.get_possible_communications(g) {
            let stage = Rc::new(Stage {
                computations_a: computations.clone(),
                computations_b: smallvec![],
                communications_a: smallvec![],
                communications_b: communications.clone(),
                acc_cost: prev_stage.acc_cost,
                prev: Some(prev_stage.clone()),
            });
            let new_state = State(new_a.clone(), new_b, stage);
            update_pareto(g, pareto, new_state);
        }
    }

    // 2. computation on b, communication on a
    for (new_b, computations) in b.get_possible_computations(g) {
        if new_b.progress() > a.progress() { // b is more advanced
            continue
        }
        for (new_a, communications) in a.get_possible_communications(g) {
            let stage = Rc::new(Stage {
                computations_a: smallvec![],
                computations_b: computations.clone(),
                communications_a: communications.clone(),
                communications_b: smallvec![],
                acc_cost: prev_stage.acc_cost,
                prev: Some(prev_stage.clone()),
            });
            let new_state = State(new_a, new_b.clone(), stage);
            update_pareto(g, pareto, new_state);
        }
    }

    // 3. Assuming this function is called in the order of costs in a single cell, such that states can updates other states in the same progress (because only small cost states may update large cost state by appending a communication-only stage that must increase the cost)
    if !prev_stage.is_overlapping() {
        return
    }

    for (new_a, communications_a) in a.get_possible_communications(g).into_iter().chain([(a.clone(), smallvec![])]) {
        for (new_b, communications_b) in b.get_possible_communications(g).into_iter().chain([(b.clone(), smallvec![])]) {
            if communications_a.is_empty() && communications_b.is_empty() {
                continue
            }
            let stage = Rc::new(Stage {
                computations_a: smallvec![],
                computations_b: smallvec![],
                communications_a: communications_a.clone(),
                communications_b: communications_b.clone(),
                acc_cost: prev_stage.acc_cost,
                prev: Some(prev_stage.clone()),
            });
            let new_state = State(new_a.clone(), new_b, stage);
            update_pareto(g, pareto, new_state);
        }
    }
    for (new_a, computations_a) in a.get_possible_computations(g).into_iter().chain([(a.clone(), smallvec![])]) {
        for (new_b, computations_b) in b.get_possible_computations(g).into_iter().chain([(b.clone(), smallvec![])]) {
            if computations_a.is_empty() && computations_b.is_empty() {
                continue
            }
            if new_b.progress() > new_a.progress() || new_b.progress() + PROGRESS_LIMIT < new_a.progress() {
                continue
            }
            let stage = Rc::new(Stage {
                computations_a: computations_a.clone(),
                computations_b: computations_b.clone(),
                communications_a: smallvec![],
                communications_b: smallvec![],
                acc_cost: prev_stage.acc_cost,
                prev: Some(prev_stage.clone()),
            });
            let new_state = State(new_a.clone(), new_b, stage);
            update_pareto(g, pareto, new_state);
        }
    }
}

fn update_pareto(g: &Graph, pareto: &mut Pareto, state: State) {
    if CTRLC_RECEIVED.load(std::sync::atomic::Ordering::Relaxed) {
        panic!("interupted")
    }
    let key = state.get_pareto_cell_key();
    let new_cost = state.2.acc_cost;
    match pareto[state.0.cut.0][state.1.cut.0].entry(key) {
        std::collections::btree_map::Entry::Vacant(x) => x.insert(state).ignore(),
        std::collections::btree_map::Entry::Occupied(mut x) if new_cost <= x.get().2.acc_cost => *x.get_mut() = state,
        _ => {}
    }
}

