use std::{rc::Rc, collections::{BTreeMap, BTreeSet}, ops::Index, cell::RefCell};

use cpython::{ToPyObject, PyObject, Python, PyList, PyDict, PyResult, PythonObject};
use oh_my_rust::*;

use crate::{graph::{NodeIndex, TensorIndex, Form, Node, Signature, SignatureIndex, Tensor, Collective}, SVec, smallvec, CTRLC_RECEIVED, profiler::Profiler};

const PROGRESS_LIMIT: usize = 10;

#[derive(Clone, Copy)]
pub struct Computation {
    node: NodeIndex,
    signature: SignatureIndex
}

#[derive(Clone, Copy)]
pub struct Communication {
    tensor: TensorIndex,
    old_form: Form,
    new_form: Form
}

impl Communication {
    pub fn collectives(&self) -> SVec<Collective, 2> {
        self.old_form.collective_reform(self.new_form).unwrap() // A Communication must be a valid transform
    }
}

#[derive(Clone, Default)]
pub struct Stage {
    computations: SVec<Computation, PROGRESS_LIMIT>,
    communications: SVec<Communication>,

    cost: Cost,
    prev: Option<Rc<Stage>>
}

impl Stage {
    // empty stage is an invalid stage serves as the prev of the first stage
    fn is_empty(&self) -> bool {
        self.computations.is_empty() && self.communications.is_empty()
    }
}

#[derive(Clone, Default, PartialEq)]
struct Cost {
    acc_time: f64,
    debt: f64
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Cost) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::{Less, Greater, Equal};
        match (self.acc_time.partial_cmp(&other.acc_time), self.debt.partial_cmp(&other.debt)) {
            (Some(Less), Some(Less | Equal)) => Some(Less),
            (Some(Less | Equal), Some(Less)) => Some(Less),
            (Some(Greater), Some(Greater | Equal)) => Some(Greater),
            (Some(Greater | Equal), Some(Greater)) => Some(Greater),
            (Some(Equal), Some(Equal)) => Some(Equal),
            _ => None
        }
    }
}

struct Cut { // the cut on a single duplica
    next_node: NodeIndex,
    state_tensors: Vec<TensorIndex>, // sorted
    state_tensors_reverse_map: BTreeMap<TensorIndex, usize>, // map a tensor to its index in the state_tensors
}

impl Cut {
    fn new(next_node: NodeIndex, state_tensors: Vec<TensorIndex>) -> Cut {
        let reverse_map = state_tensors.iter().enumerate().map(|(i, &tensor_index)| (tensor_index, i)).collect();
        Cut { next_node, state_tensors, state_tensors_reverse_map: reverse_map }
    }

    fn get_all_in_graph(g: &Graph) -> Vec<Cut> {
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
            Cut::new(NodeIndex(i), x)
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
pub struct State {
    cut: CutIndex,
    available_forms: Vec<OneOrTwoForms>, // all available forms for each state tensor, some of them may be not communicateable (tracked separately)
}

impl State {
    fn get_possible_actions(&self, g: &Graph) -> Vec<(Computation, SVec<Communication>, State)> {
        if g.n_nodes() <= g[self.cut].next_node.0 {
            return vec![]
        }

        let next_node = &g[g[self.cut].next_node];
        let mut result = vec![];

        for (signature_id, signature) in next_node.signatures.iter().enumerate() {
            let mut communications = SVec::default();

            for (&input_tensor, &input_tensor_form) in next_node.inputs.iter().zip(signature.input_forms.iter()) {
                let i = g[self.cut].state_tensors_reverse_map.get(&input_tensor).expect("input tensor not in state tensors");
                let available_forms = self.available_forms[*i];
                if available_forms.contains(input_tensor_form) {
                    continue
                }
                if available_forms.len() > 1 {
                    continue // too many forms
                }
                let available_form = available_forms.unwrap();
                if available_form.collective_reform(input_tensor_form).is_none() {
                    continue // cannot reform like this
                }
                communications.push(Communication { tensor: input_tensor, old_form: available_form, new_form: input_tensor_form })
            }

            let mut next_state = State {
                cut: self.cut + 1,
                available_forms: g[self.cut + 1].state_tensors.iter().map(|tensor_index| {
                    g[self.cut].state_tensors_reverse_map.get(tensor_index)
                        .map(|&i| {
                            let mut available_forms = self.available_forms[i];
                            if let Some(corresponding_communication) = communications.iter().find(|x| x.tensor == *tensor_index) {
                                available_forms.insert(corresponding_communication.new_form)
                            }
                            available_forms
                        })
                        .unwrap_or_else(|| {
                            let i = next_node.outputs.iter().position(|x| x == tensor_index).unwrap();
                            OneOrTwoForms::one(signature.output_forms[i])
                        })
                }).collect()
            };
            let computation = Computation { node: g[self.cut].next_node, signature: SignatureIndex(signature_id) };
            result.push((computation, communications, next_state))
        }

        result
    }
}

struct Graph<'g> {
    nodes: &'g [Node],
    tensors: &'g [Tensor],
    cuts: Vec<Cut>,
    profiler: &'g dyn Profiler
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
    type Output = Cut;

    fn index(&self, index: CutIndex) -> &Self::Output {
        &self.cuts[index.0]
    }
}

impl<'g> Graph<'g> {
    fn new(graph: &'g crate::graph::Graph, profiler: &'g dyn Profiler) -> Graph<'g> {
        let crate::graph::Graph { nodes, tensors } = graph;
        let mut result = Graph { nodes, tensors, cuts: vec![], profiler };
        result.cuts = Cut::get_all_in_graph(&result);
        result
    }

    fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    fn n_cuts(&self) -> usize {
        self.cuts.len()
    }
}

pub fn dp3(graph: &crate::graph::Graph, profiler: &dyn Profiler) {
    let g = Graph::new(graph, profiler);
    let n_cut = g.n_cuts();

    let mut pareto: Pareto = (0..n_cut).map(|_| Default::default()).collect();
    pareto[0].insert(vec![], vec![Rc::new(Stage::default())]);

    #[allow(clippy::needless_range_loop)]
    for cut_index in 0..n_cut-2 {
        info!("expanding on cut {} with {} states and {} paths", cut_index, pareto[cut_index].len(), pareto[cut_index].values().map(|x| x.len()).sum::<usize>());
        for (available_forms, prev_stages) in std::mem::take(&mut pareto[cut_index]) { // take out for eager free memory as well as to release reference to pareto
            let state = State { cut: cut_index.into(), available_forms };
            explore_next_stage(&g, &mut pareto, state, &prev_stages);
        }
    }

    let mut best_time = f64::MAX;
    let mut best_path = None;
    for (available_forms, stages) in pareto[n_cut-2].iter() {
        if available_forms[0].contains(Form::Reduce) {
            for stage in stages {
                let time = stage.cost.acc_time + stage.cost.debt;
                if best_path.is_none() || time < best_time {
                    best_time = time;
                    best_path = Some(stage)
                }
            }
            break
        }
    }
    dump_path(&g, best_path.unwrap())
}

// Pareto[cut][state_tensors] = list of pareto stages leads to it
// conceptually state -> vec<stage> but orgnized like this for better indexing & eager memory relcaim
type Pareto = Vec<BTreeMap<Vec<OneOrTwoForms>, Vec<Rc<Stage>>>>;

// explore all stages that could be appended to the state and try to add them to the pareto
fn explore_next_stage(g: &Graph, pareto: &mut Pareto, state: State, prev_stages: &[Rc<Stage>]) {
    let communicatable_tensors: BTreeSet<_> = g[state.cut].state_tensors.iter().copied().collect();
    let mut to_explore = vec![(state, SVec::<Computation, PROGRESS_LIMIT>::default(), SVec::<Communication>::default())];
    while let Some((state, computations, communications)) = to_explore.pop() {
        let forward_comp_time = computations.iter().map(|comp| g.profiler.get_computation_forward_time(&g[comp.node], comp.signature)).sum::<f64>();
        let backward_comp_time = computations.iter().map(|comp| g.profiler.get_computation_backward_time(&g[comp.node], comp.signature)).sum::<f64>();
        let forward_comm_time = communications.iter().map(|comm| g.profiler.get_communication_forward_time(g[comm.tensor].size, comm.old_form, comm.new_form)).sum::<f64>();
        let backward_comm_time = communications.iter().map(|comm| {
            let mut time = g.profiler.get_communication_backward_time(g[comm.tensor].size, comm.old_form, comm.new_form);
            if comm.old_form == Form::Replicate {
                time /= 2. // the all-reduce operation for parameters on the backward pass are not blocking, so we give them less weight
            }
            time
        }).sum::<f64>();

        for prev_stage in prev_stages.iter().cloned() {
            let new_acc_time = prev_stage.cost.acc_time +
                /*forward*/ prev_stage.cost.debt.max(forward_comm_time) + forward_comp_time.max(forward_comm_time) +
                /*backward*/ (2. * prev_stage.cost.debt).max(backward_comm_time) + backward_comp_time.max(backward_comm_time);
            // let new_acc_time = prev_stage.cost.acc_time +
            //     /*forward*/ prev_stage.cost.debt + forward_comm_time + forward_comp_time + forward_comm_time +
            //     /*backward*/ (2. * prev_stage.cost.debt) + backward_comm_time + backward_comp_time + backward_comm_time;

            let new_debt = forward_comp_time;
            let new_stage = Stage {
                computations: computations.clone(),
                communications: communications.clone(),
                cost: Cost { acc_time: new_acc_time, debt: new_debt },
                prev: Some(prev_stage),
            };
            if is_stage_valid(&new_stage) {
                update_pareto(g, pareto, &state, new_stage)
            }
        }

        if computations.len() > PROGRESS_LIMIT {
            continue
        }
        for (new_computation, new_communications, new_state) in state.get_possible_actions(g) {
            if new_communications.iter().any(|c| !communicatable_tensors.contains(&c.tensor)) {
                continue
            }
            to_explore.push((
                new_state,
                computations.clone().apply(|x| x.push(new_computation)),
                communications.clone().apply(|x| x.extend_from_slice(&new_communications))
            ))
        }
    }
}

// check if a stage is valid by
// 1. it must has computation
// 2. if it does not have communication, it must be either
//    - is the first stage,
//    - or the previous stage reaches PROGRESS_LIMIT
// this function does not check if the communications are actually communicatable. It is checked in explore_next_stage
fn is_stage_valid(stage: &Stage) -> bool {
    if stage.computations.is_empty() {
        return false
    }
    if stage.communications.is_empty() {
        let prev = stage.prev.as_ref().unwrap();
        if prev.prev.is_none() { // first stage as a empty invalid prev stage
            return true
        }
        if prev.computations.len() > PROGRESS_LIMIT {
            return true
        }
        return false
    }
    true
}

// update the pareto given a stage that can leads to a state
fn update_pareto(g: &Graph, pareto: &mut Pareto, state: &State, stage: Stage) {
    use std::cmp::Ordering::{Less, Greater, Equal};

    if CTRLC_RECEIVED.load(std::sync::atomic::Ordering::Relaxed) {
        panic!("interupted")
    }
    let key = state.available_forms.clone();
    match pareto[state.cut.0].entry(key) {
        std::collections::btree_map::Entry::Vacant(x) => x.insert(vec![Rc::new(stage)]).ignore(),
        std::collections::btree_map::Entry::Occupied(mut x) => {
            let candidates = x.get_mut();
            let mut should_add = true;
            candidates.retain(|c| {
                match c.cost.partial_cmp(&stage.cost) {
                    Some(Greater) => false, // remove the bad candidate
                    Some(Equal | Less) => { // there is a strictly better or exactly the same candidate. We keep the candidates and reject the adding
                        should_add = false;
                        true
                    }
                    None => true, // do nothing and add this to the pareto unless there are other strictly better candidates
                }
            });
            if should_add {
                candidates.push(Rc::new(stage))
            }
        },
    }
}

fn dump_path(g: &Graph, stage: &Stage) {
    if let Some(prev) = stage.prev.as_ref() {
        if !prev.is_empty() {
            dump_path(g, prev)
        }
    }
    println!("======");
    for comm in stage.communications.iter() {
        println!("Tensor {} of size {} from {:?} to {:?}", comm.tensor.0, g[comm.tensor].size, comm.old_form, comm.new_form);
    }
    for comp in stage.computations.iter() {
        let node = &g[comp.node];
        println!("Node {} ({}) {} {:?} -> {:?}", comp.node.0, node.origin_id, node.name, node.signatures[comp.signature.0].input_forms, node.signatures[comp.signature.0].output_forms);
    }
}

fn export_path(py: Python, g: &Graph, stage: &Stage) -> PyResult<PyList> {
    let prev = stage.prev.as_ref().unwrap();
    let mut py_stages = if !prev.is_empty() {
        export_path(py, g, prev)?
    } else {
        PyList::new(py, &[])
    };

    let mut py_computations = PyList::new(py, &[]);
    let mut py_communications = PyList::new(py, &[]);

    for communication in &stage.communications {
        let mut py_communication = PyDict::new(py);
        py_communication.set_item(py, "origin_node_id", {
            let producer_node = &g[g[communication.tensor].producer];
            if producer_node.companions.is_empty() {
                producer_node.origin_id
            } else {
                let i = producer_node.outputs.iter().position(|&x| x == communication.tensor).unwrap();
                producer_node.companions[i]
            }
        })?;
        py_communication.set_item(py, "old_form", communication.old_form.to_string())?;
        py_communication.set_item(py, "new_form", communication.new_form.to_string())?;
        py_communication.set_item(py, "collectives", {
            let mut py_collectives = PyList::new(py, &[]);
            for collective in communication.collectives() {
                py_collectives.append(py, collective.to_string().into_py_object(py).into_object())
            }
            py_collectives
        })?;
        py_communications.append(py, py_communication.into_object())
    }

    Ok(py_stages)
}
