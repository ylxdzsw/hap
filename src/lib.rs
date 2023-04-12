#![allow(unused)]
#![allow(non_upper_case_globals)]
#![feature(let_chains)]

use std::iter::Product;
use std::ops::{Index, IndexMut, Add, Mul, Div};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt::{Display, Debug, Formatter};
use std::cmp::Ordering;
use float_ord::FloatOrd;
use cpython::{PyResult, PyTuple, ToPyObject, ObjectProtocol, Python, PyObject, PyDict, PyClone, PyNone};
use smallvec::{SmallVec, smallvec};

pub type SVec<T, const N: usize = 1> = SmallVec<[T; N]>;

type Dimension = u8;
type Shape = SVec<usize, 4>;
type SymbolicShape = SVec<Expression, 4>;

static CTRLC_TRAPPED: AtomicBool = AtomicBool::new(false);
static CTRLC_RECEIVED: AtomicBool = AtomicBool::new(false);

static init_script: &str = r#"
import collectives
import operator
import models

def get_shape_of_param_or_buffer(graph_module, node):
    try:
        p = graph_module.get_parameter(node.target)
    except AttributeError:
        p = graph_module.get_buffer(node.target)
    return tuple(p.shape)

def split_param_or_buffer(graph_module, target, sharding_lengths, dim, rank):
    import torch

    try:
        p = graph_module.get_parameter(target)
    except AttributeError:
        p = graph_module.get_buffer(target)
    p.data = torch.split(p.data, sharding_lengths, dim)[rank]
"#;

cpython::py_module_initializer!(hetspmd, |py, m| {
    if !CTRLC_TRAPPED.load(std::sync::atomic::Ordering::Relaxed) {
        ctrlc::set_handler(|| {
            CTRLC_RECEIVED.store(true, std::sync::atomic::Ordering::Relaxed)
        }).unwrap();
    }

    m.add(py, "init", cpython::py_fn!(py, init() -> PyResult<PyNone> {
        py.run(init_script, None, None).map(|_| PyNone)
    }))?;

    m.add(py, "main", cpython::py_fn!(py, main(py_graph_module: PyObject, py_config: PyObject) -> PyResult<PyObject> {
        macro_rules! get_config {
            ($key: expr) => { py_config.get_item(py, $key)?.extract(py)? }
        }

        let py_input_shape_dict = py_config.get_item(py, "input_shape").unwrap();
        let rgraph = load_fx_graph(py, py_graph_module.clone_ref(py), py_input_shape_dict)?;

        // eprintln!("graph: {rgraph:#?}");

        let mut triples = analyze_rgraph(&rgraph);
        let mut default_properties = vec![];

        heuristics::unique_computation(&mut triples, &mut default_properties);
        heuristics::unique_communication(&mut triples, &mut default_properties);
        heuristics::fuse_free_triple(&mut triples, &mut default_properties);
        heuristics::fuse_communication(&mut triples, &mut default_properties);

        // for triple in triples.iter() {
        //     eprintln!("{triple}");
        // }

        let cluster_info = ClusterInfo {
            device_flops: get_config!("device_flops"),
            all_reduce_bandwidth: get_config!("all_reduce_bandwidth"),
            all_gather_bandwidth: get_config!("all_gather_bandwidth"),
            reduce_scatter_bandwidth: get_config!("reduce_scatter_bandwidth"),
            all_to_all_bandwidth: get_config!("all_to_all_bandwidth")
        };

        // todo: merge into context?
        let profiler = Profiler {
            rgraph: &rgraph,
            cluster_info: &cluster_info
        };

        let triple_set = IndexedHoareTripleSet::new(triples);

        let symbol_id_counter = AtomicUsize::new(0);
        let sharding_ratios = (0..rgraph.n_segments).map(|s| {
            (0..cluster_info.n_devices()).map(|d| {
                SymbolId(symbol_id_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();

        let computation_power_sum = cluster_info.device_flops.iter().sum::<f64>();
        let computation_power_ratio = cluster_info.device_flops.iter().map(|f| f / computation_power_sum).collect::<Vec<_>>();
        let mut symbol_values = vec![0.; symbol_id_counter.load(std::sync::atomic::Ordering::Relaxed)];
        for segment_sharding_ratio in sharding_ratios.iter() {
            for (sharding_ratio, computation_power_ratio) in segment_sharding_ratio.iter().zip(computation_power_ratio.iter()) {
                symbol_values[sharding_ratio.0] = *computation_power_ratio;
            }
        }

        let sharding_ratios_exp = sharding_ratios.iter().map(|s| {
            s.iter().map(|d| Expression::symbol(*d)).collect()
        }).collect::<Vec<_>>();

        let a_star_context = AStarContext {
            triple_set: &triple_set,
            sharding_ratios: &sharding_ratios_exp,
            symbol_values: &symbol_values
        };

        let best_program = a_star(&a_star_context, &default_properties, &profiler);
        eprintln!("===== Result =====\n\n");
        best_program.show(&triple_set);

        sharding_ratio_optimization(&best_program, &triple_set, &sharding_ratios, &profiler);





        let mut codegen_context = CodegenContext::new(
            py, py_graph_module, &rgraph, get_config!("rank"),
            sharding_ratios.iter().map(|s| {
                s.iter().map(|d| Expression::symbol(*d).instantialize(&symbol_values)).collect()
            }).collect()
        )?;

        best_program.codegen(&triple_set, &mut codegen_context)?;

        Ok(codegen_context.graph)
    }))?;

    Ok(())
});

macro_rules! new_usize_type {
    ($visibility: vis, $type_name: ident) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
        #[repr(transparent)]
        $visibility struct $type_name(pub usize);
        impl<T: Into<$type_name>> std::ops::Add<T> for $type_name {
            type Output = $type_name;

            fn add(self, rhs: T) -> $type_name {
                $type_name(self.0 + rhs.into().0)
            }
        }

        impl<T: Into<$type_name>> std::ops::AddAssign<T> for $type_name {
            fn add_assign(&mut self, rhs: T) {
                self.0 += rhs.into().0;
            }
        }

        impl From<usize> for $type_name {
            fn from(x: usize) -> $type_name {
                $type_name(x)
            }
        }

        impl std::fmt::Display for $type_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    }
}

pub(crate) use new_usize_type;

macro_rules! py_dict {
    ($py:expr, $($key:ident => $value:expr),*) => {{
        let dict = PyDict::new($py);
        $(
            dict.set_item($py, stringify!($key), &$value).unwrap();
        )*
        dict
    }}
}

// Some names:
// R stands for Reference (the nodes and tensors in the orignal single card graph)
// D stands for Distributed (the nodes and tensors in the SIMD graph)
// Op is a curried operator with non-tensor parameters filled in
// Parameters are the parameters of the model. "Attr" is only used in "GetAttr" to keep the same as PyTorch.
// Placeholders are the inputs to the model
// The "input" of an instruction is the tensor that is read by the instruction
new_usize_type!(pub, RNodeId);
new_usize_type!(pub, RTensorId);
new_usize_type!(pub, DNodeId);
new_usize_type!(pub, DTensorId);
new_usize_type!(pub, SegmentId);

pub struct HoareTriple {
    pre_conditions: SVec<Property, 4>,
    post_conditions: SVec<Property>,
    negative_post_conditions: Vec<Property>,
    instruction: String, // for debugging purpose
    codegen: Rc<dyn Fn(&mut CodegenContext) -> PyResult<()>>,
    profile: Rc<dyn Fn(&mut ProfileContext) -> (Profile, Profile)>
}

impl Display for HoareTriple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{")?;
        for (i, p) in self.pre_conditions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{p}")?;
        }
        write!(f, "}} {} {{", self.instruction)?;
        for (i, p) in self.post_conditions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{p}")?;
        }
        // for (i, p) in self.negative_post_conditions.iter().enumerate() {
        //     if i > 0 || !self.post_conditions.is_empty() {
        //         write!(f, ", ")?;
        //     }
        //     write!(f, "¬({p})")?;
        // }
        write!(f, "}}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Property {
    HasTensor(RTensorId, TensorRelation),
    Finished,

    AllowCommunication(RTensorId),
    AllowComputation(RTensorId), // also includes placeholder and getattr
}

impl Display for Property {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Property::HasTensor(tensor_id, relation) => {
                write!(f, "{}|{:?}", tensor_id, relation)
            }
            Property::Finished => write!(f, "finished"),
            Property::AllowCommunication(tensor_id) => {
                write!(f, "{}|allow_communication", tensor_id)
            }
            Property::AllowComputation(tensor_id) => {
                write!(f, "{}|allow_computation", tensor_id)
            }
        }
    }
}

impl Property {
    fn identity(tensor_id: RTensorId) -> Property {
        Property::HasTensor(tensor_id, TensorRelation::Identity)
    }

    fn gather(tensor_id: RTensorId, dim: Dimension) -> Property {
        Property::HasTensor(tensor_id, TensorRelation::Gather(dim))
    }

    fn reduce(tensor_id: RTensorId) -> Property {
        Property::HasTensor(tensor_id, TensorRelation::Reduce)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorRelation {
    Gather(Dimension),
    Reduce,
    Identity,
}

impl HoareTriple {
    fn get_cost_symbolic(&self, profiler: &Profiler, sharding_ratios: &[Vec<Expression>]) -> (Vec<Expression>, Vec<Expression>) {
        let (computation_times, communication_times): (Vec<_>, Vec<_>) = (0..profiler.cluster_info.n_devices()).map(|i| {
            // TODO: a lot of unnecessary work. Need benchmark.
            let mut profile_context = ProfileContext {
                profiler,
                sharding_ratios,
                device_index: i,
            };
            let (forward, backward) = (self.profile)(&mut profile_context);
            let computation_time = (forward.flops + backward.flops) / profiler.cluster_info.device_flops[i];
            let communication_time =
                (forward.all_gather + backward.all_gather) / profiler.cluster_info.all_gather_bandwidth +
                (forward.all_reduce + backward.all_reduce) / profiler.cluster_info.all_reduce_bandwidth +
                (forward.reduce_scatter + backward.reduce_scatter) / profiler.cluster_info.reduce_scatter_bandwidth +
                (forward.all_to_all + backward.all_to_all) / profiler.cluster_info.all_to_all_bandwidth;

            (computation_time, communication_time)
        }).unzip();

        (computation_times, communication_times)
    }

    fn get_cost(&self, profiler: &Profiler, sharding_ratios: &[Vec<Expression>], symbol_values: &[f64]) -> f64 {
        // idea: may cache the cost of each triple while using the same sharding ratios
        let (computation_times, communication_times) = self.get_cost_symbolic(profiler, sharding_ratios);

        let computation_time = computation_times.into_iter().map(|x| x.instantialize(symbol_values)).map(FloatOrd).max().unwrap().0;
        let communication_time = communication_times.into_iter().map(|x| x.instantialize(symbol_values)).map(FloatOrd).max().unwrap().0;

        computation_time + communication_time
    }

    fn fuse_into(&self, consumer: &HoareTriple) -> HoareTriple {
        // note: must avoid circles. Currently I only fuse free tensors and communications, which are safe

        for negative_post_condition in self.negative_post_conditions.iter() {
            assert!(!consumer.pre_conditions.contains(&negative_post_condition));
        }

        let pre_conditions = self.pre_conditions.iter().chain(
            consumer.pre_conditions.iter()
                .filter(|c| !self.pre_conditions.contains(c) && !self.post_conditions.contains(c))
        ).cloned().collect();

        let post_conditions = self.post_conditions.iter().chain(consumer.post_conditions.iter()).cloned().collect();

        let negative_post_conditions = self.negative_post_conditions.iter().chain(consumer.negative_post_conditions.iter()).cloned().collect();

        let instruction = format!("{}, {}", self.instruction, consumer.instruction);

        let codegen = {
            let self_codegen = self.codegen.clone();
            let consumer_codegen = consumer.codegen.clone();
            Rc::new(move |ctx: &mut CodegenContext| {
                self_codegen(ctx)?;
                consumer_codegen(ctx)
            })
        };

        let profile = {
            let self_profile = self.profile.clone();
            let consumer_profile = consumer.profile.clone();
            Rc::new(move |ctx: &mut ProfileContext| {
                let (forward1, backward1) = self_profile(ctx);
                let (forward2, backward2) = consumer_profile(ctx);
                (forward1 + forward2, backward1 + backward2)
            })
        };

        HoareTriple {
            pre_conditions,
            post_conditions,
            negative_post_conditions,
            instruction,
            codegen,
            profile
        }
    }
}

#[derive(Debug, Clone)]
struct Profiler<'r, 'c> {
    rgraph: &'r RGraph,
    cluster_info: &'c ClusterInfo,
}

#[derive(Debug, Clone, Default)]
struct Profile {
    flops: Expression,
    all_reduce: Expression,
    all_gather: Expression,
    all_to_all: Expression,
    reduce_scatter: Expression,
}

impl Add for Profile {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Profile {
            flops: self.flops + rhs.flops,
            all_reduce: self.all_reduce + rhs.all_reduce,
            all_gather: self.all_gather + rhs.all_gather,
            all_to_all: self.all_to_all + rhs.all_to_all,
            reduce_scatter: self.reduce_scatter + rhs.reduce_scatter,
        }
    }
}

struct ProfileContext<'p, 's, 'r, 'c> {
    profiler: &'p Profiler<'r, 'c>,
    sharding_ratios: &'s [Vec<Expression>],
    device_index: usize
}

impl<'p, 's, 'r, 'c> ProfileContext<'p, 's, 'r, 'c> {
    fn get_shape_by_property(&self, property: Property) -> SymbolicShape {
        if let Property::HasTensor(tensor_id, rel) = property {
            let tensor = &self.profiler.rgraph[tensor_id];
            match rel {
                TensorRelation::Identity | TensorRelation::Reduce => tensor.shape.iter().map(|s| Expression::constant(*s as _)).collect(),
                TensorRelation::Gather(dim) => {
                    let dim = dim as usize;
                    let mut shape: SymbolicShape = tensor.shape.iter().map(|s| Expression::constant(*s as _)).collect();
                    shape[dim] = sharding_symbolic(tensor.shape[dim], &self.sharding_ratios[tensor.segment_id.0])[self.device_index].clone(); // unnecessary clone
                    shape
                }
            }
        } else {
            unreachable!()
        }
    }
}

#[derive(Default, Debug, Clone)]
struct Program {
    triple_ids: Vec<HoareTripleId>,
    properties: BTreeSet<Property>,
    cost: f64,
    ecost: f64,
}

impl Program {
    fn empty(properties: impl IntoIterator<Item=Property>) -> Program {
        Program { properties: properties.into_iter().collect(), ..Default::default() }
    }

    fn with_a_new_triple(&self, ctx: &AStarContext, triple_id: HoareTripleId, profiler: &Profiler) -> Program {
        let mut triples = self.triple_ids.clone();
        triples.push(triple_id);

        let triple = &ctx.triple_set[triple_id];

        let mut properties = self.properties.iter()
            .filter(|p| !triple.negative_post_conditions.contains(p))
            .chain(triple.post_conditions.iter())
            .cloned()
            .collect();

        remove_irrelavent_properties(&mut properties, &ctx.triple_set);

        let cost = self.cost + triple.get_cost(profiler, &ctx.sharding_ratios, &ctx.symbol_values);
        let ecost = 0.0;

        Program { triple_ids: triples, properties, cost, ecost }
    }

    fn find_available_triples<'s, 't: 's>(&'s self, triple_set: &'t IndexedHoareTripleSet) -> Vec<HoareTripleId> {
        let candidates: BTreeSet<_> = self.properties.iter().flat_map(|p| triple_set.get_triples_with_pre_condition(*p)).copied().collect();

        candidates.into_iter().filter(|triple_id| {
            let triple = &triple_set[*triple_id];
            triple.pre_conditions.iter().all(|p| self.properties.contains(p)) && triple.post_conditions.iter().any(|p| !self.properties.contains(p))
        }).collect()
    }

    fn is_complete(&self) -> bool {
        self.properties.iter().any(|p| *p == Property::Finished)
    }

    fn show(&self, triple_set: &IndexedHoareTripleSet) {
        eprintln!("length: {}, cost: {}, ecost: {}", self.triple_ids.len(), self.cost, self.ecost);

        eprintln!("=== active properties ===");
        for property in &self.properties {
            eprintln!("{property}");
        }
        eprintln!("=== triples ===");
        for triple_id in &self.triple_ids {
            eprintln!("{}", triple_set[*triple_id]);
        }
    }

    fn codegen<'py, 'r>(&self, triple_set: &IndexedHoareTripleSet, ctx: &mut CodegenContext<'py, 'r>) -> PyResult<()> {
        for triple_id in &self.triple_ids {
            let triple = &triple_set[*triple_id];
            (triple.codegen)(ctx)?;
            for property in &triple.post_conditions {
                if let Property::HasTensor(_, _) = property {
                    assert!(ctx.property_implementation.contains_key(property), "{} {}", triple, property);
                }
            }
        }
        Ok(())
    }
}

// not all irrelavent properties are removed: we only remove those can be checked without recursion to speeds up this function
fn remove_irrelavent_properties(properties: &mut BTreeSet<Property>, triple_set: &IndexedHoareTripleSet) {
    let irrelavent: Vec<_> = properties.iter().filter(|property| {
        if property == &&Property::Finished {
            return false;
        }

        // sufficient but not necessary
        triple_set.get_triples_with_pre_condition(**property).iter().all(|triple_id| {
            triple_set[*triple_id].pre_conditions.iter().any(|p| {
                !properties.contains(p) && triple_set.get_triples_with_post_condition(*p).is_empty()
            })
        })
    }).cloned().collect();

    for property in irrelavent {
        properties.remove(&property);
    }
}

#[derive(Debug, Clone)]
struct ProgramHeapEntry {
    program: Program,
    total_cost: FloatOrd<f64>,
}

impl Ord for ProgramHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.total_cost.partial_cmp(&other.total_cost).unwrap()
    }
}

impl PartialOrd for ProgramHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other).reverse()) // reverse to convert the max heap to min heap
    }
}

impl Eq for ProgramHeapEntry {}

impl PartialEq for ProgramHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.total_cost == other.total_cost
    }
}

impl ProgramHeapEntry {
    fn new(program: Program) -> Self {
        let total_cost = FloatOrd(program.cost + program.ecost);
        ProgramHeapEntry { program, total_cost }
    }
}

/// a helper struct to print the iteration count and the elapsed time
struct Ticker {
    iter_count: usize,
    iter_per_print: usize,
    start_time: std::time::Instant,
}

impl Ticker {
    fn new(iter_per_print: usize) -> Self {
        Ticker { iter_count: 0, iter_per_print, start_time: std::time::Instant::now() }
    }

    fn tick(&mut self) {
        self.iter_count += 1;
        if self.iter_count % self.iter_per_print == 0 {
            eprintln!("{self}")
        }
    }
}

impl Display for Ticker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "iter count: {}, speed: {} iter/s",
            self.iter_count,
            self.iter_count as f64 / self.start_time.elapsed().as_secs_f64()
        )
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        eprintln!("{self}")
    }
}

struct AStarContext<'t, 's, 'v> {
    triple_set: &'t IndexedHoareTripleSet,
    sharding_ratios: &'s [Vec<Expression>],
    symbol_values: &'v [f64]
}

new_usize_type!(pub, HoareTripleId);

struct IndexedHoareTripleSet {
    triples: Vec<HoareTriple>,
    pre_condition_index: BTreeMap<Property, Vec<HoareTripleId>>,
    post_condition_index: BTreeMap<Property, Vec<HoareTripleId>>,
}

impl IndexedHoareTripleSet {
    fn new(triples: Vec<HoareTriple>) -> Self {
        let mut pre_condition_index: BTreeMap<Property, Vec<HoareTripleId>> = Default::default();
        let mut post_condition_index: BTreeMap<Property, Vec<HoareTripleId>> = Default::default();

        for (i, triple) in triples.iter().enumerate() {
            for p in &triple.pre_conditions {
                pre_condition_index.entry(*p).or_default().push(HoareTripleId(i));
            }
            for p in &triple.post_conditions {
                post_condition_index.entry(*p).or_default().push(HoareTripleId(i));
            }
        }

        IndexedHoareTripleSet { triples, pre_condition_index, post_condition_index }
    }

    fn get_triples_with_pre_condition(&self, property: Property) -> &[HoareTripleId] {
        static empty: Vec<HoareTripleId> = vec![];
        self.pre_condition_index.get(&property).unwrap_or(&empty)
    }

    fn get_triples_with_post_condition(&self, property: Property) -> &[HoareTripleId] {
        static empty: Vec<HoareTripleId> = vec![];
        self.post_condition_index.get(&property).unwrap_or(&empty)
    }
}

impl Index<HoareTripleId> for IndexedHoareTripleSet {
    type Output = HoareTriple;

    fn index(&self, index: HoareTripleId) -> &Self::Output {
        &self.triples[index.0]
    }
}

fn a_star(ctx: &AStarContext, initial_properties: &[Property], profiler: &Profiler) -> Program {
    let mut heap = BinaryHeap::new();
    let mut best_program: Option<Program> = None;
    let mut property_cache: BTreeMap<BTreeSet<Property>, f64> = BTreeMap::new();

    heap.push(ProgramHeapEntry::new(Program::empty(initial_properties.iter().cloned())));
    property_cache.insert(initial_properties.iter().cloned().collect(), 0.);

    let mut ticker = Ticker::new(5000);

    while let Some(ProgramHeapEntry { program, .. }) = heap.pop() {
        if CTRLC_RECEIVED.load(std::sync::atomic::Ordering::Relaxed) {
            panic!("interupted")
        }

        if best_program.as_ref().map(|p| p.cost < program.cost).unwrap_or(false) {
            continue;
        }

        if let Some(&cached_cost) = property_cache.get(&program.properties) && cached_cost < program.cost { // it has been superseded by a better program
            continue;
        }

        // if ticker.iter_count % 5000 == 0 {
        //     eprintln!("{program}");
        // }

        if program.is_complete() {
            if best_program.as_ref().map(|p| p.cost > program.cost).unwrap_or(true) {
                best_program = Some(program);
            }
        } else {
            for triple_id in program.find_available_triples(&ctx.triple_set) {
                let new_program = program.with_a_new_triple(ctx, triple_id, profiler);
                if let Some(&cached_cost) = property_cache.get(&new_program.properties) && cached_cost <= new_program.cost {
                    continue
                }
                property_cache.insert(new_program.properties.clone(), new_program.cost);
                heap.push(ProgramHeapEntry::new(new_program));
            }
        }

        ticker.tick();
    }

    best_program.unwrap()
}

#[derive(Debug, Default)]
pub struct RGraph {
    nodes: Vec<RNode>,
    tensors: Vec<RTensor>,
    n_segments: usize,
}

#[derive(Debug)]
pub struct RNode {
    inputs: SVec<RTensorId, 4>,
    outputs: SVec<RTensorId>,
    instruction: RInstruction,
}

// An instruction in the reference graph without the input and output information
#[derive(Debug, Clone)]
pub enum RInstruction {
    Op(Rc<Op>),
    GetAttr(String),
    Placeholder(String),
    Output
}

#[derive(Debug)]
pub struct RTensor {
    producer: RNodeId,
    consumers: SVec<RNodeId>,

    segment_id: SegmentId,
    shape: Shape,
    communicatable: bool, // hints automatically generated for certain operatios (outputs of adaptive nodes are not communicatble), can be override by user annotation
}

impl RTensor {
    fn n_dims(&self) -> Dimension {
        self.shape.len() as _
    }

    fn size(&self) -> f64 {
        self.shape.iter().map(|x| *x as f64).product()
    }
}

impl Index<RNodeId> for RGraph {
    type Output = RNode;

    fn index(&self, index: RNodeId) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl IndexMut<RNodeId> for RGraph {
    fn index_mut(&mut self, index: RNodeId) -> &mut Self::Output {
        &mut self.nodes[index.0]
    }
}

impl Index<RTensorId> for RGraph {
    type Output = RTensor;

    fn index(&self, index: RTensorId) -> &Self::Output {
        &self.tensors[index.0]
    }
}

impl IndexMut<RTensorId> for RGraph {
    fn index_mut(&mut self, index: RTensorId) -> &mut Self::Output {
        &mut self.tensors[index.0]
    }
}

struct CodegenContext<'py, 'r> {
    py: Python<'py>,
    graph: PyObject,
    module: PyObject, // the graph_module to modify (the parameters will be sharded inplace)
    rgraph: &'r RGraph, // only to provide shape information

    rank: usize,
    sharding_ratios: Vec<Vec<f64>>,

    property_implementation: BTreeMap<Property, PyObject>
}

impl<'py, 'r> CodegenContext<'py, 'r> {
    fn new(py: Python<'py>, module: PyObject, rgraph: &'r RGraph, rank: usize, sharding_ratios: Vec<Vec<f64>>) -> PyResult<Self> {
        let graph = py.eval("torch.fx.Graph()", None, None)?;

        Ok(Self {
            py, graph, module, rgraph, rank, sharding_ratios,
            property_implementation: BTreeMap::new()
        })
    }

    fn get_property_implementation(&mut self, property: Property) -> PyObject {
        self.property_implementation[&property].clone_ref(self.py)
    }

    fn set_property_implementation(&mut self, property: Property, tensor: PyObject) {
        assert!(self.property_implementation.insert(property, tensor).is_none())
    }

    fn fx_placeholder(&mut self, placeholder_name: &str) -> PyResult<PyObject> {
        self.graph.call_method(self.py, "placeholder", (placeholder_name, ), None)
    }

    fn fx_get_attr(&mut self, parameter_name: &str) -> PyResult<PyObject> {
        self.graph.call_method(self.py, "get_attr", (parameter_name, ), None)
    }

    fn fx_call_function(&mut self, function_name: &str, args: impl ToPyObject<ObjectType = PyTuple>, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let py_function = self.py.eval(function_name, None, None)?;
        self.graph.call_method(self.py, "call_function", (py_function, args, kwargs), None)
    }

    fn fx_call_method(&mut self, method_name: &str, args: impl ToPyObject<ObjectType = PyTuple>, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        self.graph.call_method(self.py, "call_method", (method_name, args, kwargs), None)
    }

    fn fx_output(&mut self, output: PyObject) -> PyResult<PyObject> {
        self.graph.call_method(self.py, "output", (output, ), None)
    }

    fn get_shape_by_property(&self, property: Property) -> Shape {
        if let Property::HasTensor(tensor_id, rel) = property {
            let tensor = &self.rgraph[tensor_id];
            match rel {
                TensorRelation::Identity | TensorRelation::Reduce => tensor.shape.clone(),
                TensorRelation::Gather(dim) => {
                    let dim = dim as usize;
                    let mut shape = tensor.shape.clone();
                    shape[dim] = sharding_round(shape[dim], &self.sharding_ratios[tensor.segment_id.0])[self.rank];
                    shape
                }
            }
        } else {
            unreachable!()
        }
    }

    fn shard_inplace(&self, name: &str, sharding_lengths: &[usize], dim: Dimension) -> PyResult<()> {
        self.py.run("split_param_or_buffer(graph_module, target, sharding_lengths, dim, rank)", None, Some(&py_dict!(self.py,
            graph_module => self.module,
            target => name,
            sharding_lengths => sharding_lengths,
            dim => dim,
            rank => self.rank
        )))
    }
}

pub struct Op {
    py_name: String,
    codegen: Box<dyn Fn(Python, &PyObject, &[PyObject]) -> PyResult<SVec<PyObject, 1>>>,
    flops: Box<dyn Fn(&[SymbolicShape]) -> Expression>,
    info: BTreeMap<String, String>, // additional info for generating triples
}

impl Debug for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Op")
            .field("py_name", &self.py_name)
            .finish()
    }
}

struct ParserContext<'py, 'g, 's, 'r> {
    py: Python<'py>,
    graph: &'g mut RGraph,
    current_segment: &'s mut SegmentId,
    results: &'r mut Vec<Option<EvalResult>>
}

#[derive(Debug, Clone)]
enum EvalResult {
    Tensor(RTensorId),
    Tuple(SVec<RTensorId>),
}

impl EvalResult {
    fn as_tensor(&self) -> RTensorId {
        match self {
            EvalResult::Tensor(id) => *id,
            EvalResult::Tuple(_) => panic!("not a tensor")
        }
    }

    fn as_tuple(&self) -> &[RTensorId] {
        match self {
            EvalResult::Tensor(_) => panic!("not a tuple"),
            EvalResult::Tuple(ids) => ids
        }
    }
}

fn initialize_parsing_handlers(py: Python) -> PyResult<BTreeMap<*mut (), &'static dyn Fn(ParserContext, PyObject) -> PyResult<()>>> {
    let mut parsing_handlers: BTreeMap<*mut (), &'static dyn Fn(ParserContext, PyObject) -> PyResult<()>> = BTreeMap::new();
    let tensor_class = py.eval("torch.Tensor", None, None)?;

    parsing_handlers.insert(py.eval("torch.nn.functional.linear", None, None)?.as_ptr() as _, &handle_linear);
    fn handle_linear(ctx: ParserContext, py_node: PyObject) -> PyResult<()> {
        let py_id: usize = py_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract(ctx.py)?;

        let py_input_input_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "input")?;
        let py_input_weight_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "weight")?;
        let py_input_bias_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "bias")?;

        let py_input_input_id = py_input_input_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?;
        let py_input_weight_id = py_input_weight_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?;
        let py_input_bias_id = py_input_bias_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?;

        let input_input_tensor_id = ctx.results[py_input_input_id].as_ref().unwrap().as_tensor();
        let input_weight_tensor_id = ctx.results[py_input_weight_id].as_ref().unwrap().as_tensor();
        let input_bias_tensor_id = ctx.results[py_input_bias_id].as_ref().unwrap().as_tensor();

        let input_input_tensor = &ctx.graph[input_input_tensor_id];
        let input_weight_tensor = &ctx.graph[input_weight_tensor_id];
        let input_bias_tensor = &ctx.graph[input_bias_tensor_id];

        let output_shape = match &input_input_tensor.shape[..] {
            [leading_dims @ .., input_features] => {
                let output_features = input_weight_tensor.shape[0];
                assert_eq!(&input_weight_tensor.shape[..], &[output_features, *input_features]);
                assert_eq!(&input_bias_tensor.shape[..], &[output_features]);
                [leading_dims, &[output_features]].concat()
            },
            _ => panic!("invalid input shape")
        };

        let node_id = RNodeId(ctx.graph.nodes.len());
        let tensor_id = RTensorId(ctx.graph.tensors.len());

        let op = Rc::new(Op {
            py_name: "torch.nn.functional.linear".to_string(),
            codegen: Box::new(|py, graph, inputs| {
                if let [input, weight, bias] = inputs {
                    let output = graph.call_method(py, "call_function", (py.eval("torch.nn.functional.linear", None, None)?, (input, weight, bias)), None)?;
                    Ok(smallvec![output])
                } else {
                    unreachable!()
                }
            }),
            flops: Box::new(|shapes| {
                if let [input_shape, weight_shape, _bias_shape] = shapes {
                    3. * input_shape.iter().cloned().product::<Expression>() * weight_shape[0].clone()
                } else {
                    unreachable!()
                }
            }),
            info: BTreeMap::new()
        });

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            segment_id: *ctx.current_segment,
            shape: output_shape.clone().into(),
            communicatable: true
        });

        ctx.graph.nodes.push(RNode {
            inputs: smallvec![input_input_tensor_id, input_weight_tensor_id, input_bias_tensor_id],
            outputs: smallvec![tensor_id],
            instruction: RInstruction::Op(op)
        });

        ctx.graph[input_input_tensor_id].consumers.push(node_id);
        ctx.graph[input_weight_tensor_id].consumers.push(node_id);
        ctx.graph[input_bias_tensor_id].consumers.push(node_id);

        ctx.results[py_id] = Some(EvalResult::Tensor(tensor_id));

        Ok(())
    }

    parsing_handlers.insert(py.eval("torch.sigmoid", None, None)?.as_ptr() as _, &handle_sigmoid);
    fn handle_sigmoid(ctx: ParserContext, py_node: PyObject) -> PyResult<()> {
        let py_id: usize = py_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract(ctx.py)?;

        let py_input_input_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "input")?;
        let py_input_input_id = py_input_input_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?;
        let input_input_tensor_id = ctx.results[py_input_input_id].as_ref().unwrap().as_tensor();
        let input_input_tensor = &ctx.graph[input_input_tensor_id];

        let node_id = RNodeId(ctx.graph.nodes.len());
        let tensor_id = RTensorId(ctx.graph.tensors.len());

        let op = Rc::new(Op {
            py_name: "torch.sigmoid".to_string(),
            codegen: Box::new(|py, graph, inputs| {
                let input = &inputs[0];
                let output = graph.call_method(py, "call_function", (py.eval("torch.sigmoid", None, None)?, (input, )), None)?;
                Ok(smallvec![output])
            }),
            flops: Box::new(|shapes| {
                let input_shape = &shapes[0];
                3. * input_shape.iter().cloned().product::<Expression>()
            }),
            info: BTreeMap::new()
        });

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            segment_id: *ctx.current_segment,
            shape: input_input_tensor.shape.clone(),
            communicatable: false
        });

        ctx.graph.nodes.push(RNode {
            inputs: smallvec![input_input_tensor_id],
            outputs: smallvec![tensor_id],
            instruction: RInstruction::Op(op)
        });

        ctx.graph[input_input_tensor_id].consumers.push(node_id);
        ctx.results[py_id] = Some(EvalResult::Tensor(tensor_id));
        Ok(())
    }

    parsing_handlers.insert(py.eval("torch.sum", None, None)?.as_ptr() as _, &handle_sum);
    fn handle_sum(ctx: ParserContext, py_node: PyObject) -> PyResult<()> {
        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "dim").map(|x| x.is_none(ctx.py)).unwrap_or(true));
        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "keepdim").map(|x| x.is_none(ctx.py)).unwrap_or(true));
        assert!(py_node.getattr(ctx.py, "args")?.len(ctx.py)? == 1);

        let py_id: usize = py_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract(ctx.py)?;

        let py_input_input_node = py_node.getattr(ctx.py, "args")?.get_item(ctx.py, 0)?;
        let py_input_input_id = py_input_input_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?;
        let input_input_tensor_id = ctx.results[py_input_input_id].as_ref().unwrap().as_tensor();

        let node_id = RNodeId(ctx.graph.nodes.len());
        let tensor_id = RTensorId(ctx.graph.tensors.len());

        let op = Rc::new(Op {
            py_name: "torch.sum".to_string(),
            codegen: Box::new(|py, graph, inputs| {
                let input = &inputs[0];
                let output = graph.call_method(py, "call_function", (py.eval("torch.sum", None, None)?, (input, )), None)?;
                Ok(smallvec![output])
            }),
            flops: Box::new(|input_shapes| {
                let input_shape = &input_shapes[0];
                input_shape.iter().cloned().product::<Expression>()
            }),
            info: BTreeMap::new()
        });

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            segment_id: *ctx.current_segment,
            shape: smallvec![],
            communicatable: false
        });

        ctx.graph.nodes.push(RNode {
            inputs: smallvec![input_input_tensor_id],
            outputs: smallvec![tensor_id],
            instruction: RInstruction::Op(op)
        });

        ctx.graph[input_input_tensor_id].consumers.push(node_id);
        ctx.results[py_id] = Some(EvalResult::Tensor(tensor_id));
        Ok(())
    }

    parsing_handlers.insert(py.eval("models.new_segment", None, None)?.as_ptr() as _, &handle_new_segment);
    fn handle_new_segment(ctx: ParserContext, py_node: PyObject) -> PyResult<()> {
        assert!(py_node.getattr(ctx.py, "args")?.len(ctx.py)? == 0);

        let py_id: usize = py_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract(ctx.py)?;

        let py_input_input_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "x")?;
        let py_input_input_id = py_input_input_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?;
        let input_input_tensor_id = ctx.results[py_input_input_id].as_ref().unwrap().as_tensor();

        *ctx.current_segment += 1;

        // TODO: somehow forbid sharding it, or we need to insert communication

        ctx.results[py_id] = Some(EvalResult::Tensor(input_input_tensor_id));
        Ok(())
    }

    parsing_handlers.insert(tensor_class.getattr(py, "transpose")?.as_ptr() as _, &handle_transpose);
    fn handle_transpose(ctx: ParserContext, py_node: PyObject) -> PyResult<()> {
        let py_id: usize = py_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract(ctx.py)?;

        let py_input_node = py_node.getattr(ctx.py, "args")?.get_item(ctx.py, 0)?;
        let mut dim0 = py_node.getattr(ctx.py, "args")?.get_item(ctx.py, 1)?.extract::<i32>(ctx.py)?;
        let mut dim1 = py_node.getattr(ctx.py, "args")?.get_item(ctx.py, 2)?.extract::<i32>(ctx.py)?;

        let input_tensor_id = ctx.results[py_input_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let input_shape = &ctx.graph[input_tensor_id].shape;

        if dim0 < 0 {
            dim0 += input_shape.len() as i32;
        }
        if dim1 < 0 {
            dim1 += input_shape.len() as i32;
        }

        let mut output_shape = input_shape.clone();
        output_shape.swap(dim0 as usize, dim1 as usize);

        let node_id = RNodeId(ctx.graph.nodes.len());
        let tensor_id = RTensorId(ctx.graph.tensors.len());

        let op = Rc::new(Op {
            py_name: "torch.transpose".to_string(),
            codegen: Box::new(move |py, graph, inputs| {
                let input = &inputs[0];
                let output = graph.call_method(py, "call_function", (py.eval("torch.transpose", None, None)?, (input, dim0, dim1)), None)?;
                Ok(smallvec![output])
            }),
            flops: Box::new(|input_shapes| {
                let input_shape = &input_shapes[0];
                input_shape.iter().cloned().product::<Expression>()
            }),
            info: [("dim0".to_string(), dim0.to_string()), ("dim1".to_string(), dim1.to_string())].into_iter().collect()
        });

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            segment_id: *ctx.current_segment,
            shape: output_shape,
            communicatable: false
        });

        ctx.graph.nodes.push(RNode {
            inputs: smallvec![input_tensor_id],
            outputs: smallvec![tensor_id],
            instruction: RInstruction::Op(op)
        });

        ctx.graph[input_tensor_id].consumers.push(node_id);
        ctx.results[py_id] = Some(EvalResult::Tensor(tensor_id));

        Ok(())
    }

    parsing_handlers.insert(py.eval("torch.nn.functional.multi_head_attention_forward", None, None)?.as_ptr() as _, &handle_multi_head_attention_forward);
    #[allow(non_snake_case)]
    fn handle_multi_head_attention_forward(ctx: ParserContext, py_node: PyObject) -> PyResult<()> {
        let py_id: usize = py_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract(ctx.py)?;

        let py_query_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "query")?;
        let py_key_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "key")?;
        let py_value_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "value")?;
        let py_in_proj_weight_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "in_proj_weight")?;
        let py_in_proj_bias_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "in_proj_bias")?;
        let py_out_proj_weight_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "out_proj_weight")?;
        let py_out_proj_bias_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "out_proj_bias")?;
        let py_attn_mask_node = py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "attn_mask")?;


        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "bias_k").map(|x| x.is_none(ctx.py)).unwrap_or(false));
        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "bias_v").map(|x| x.is_none(ctx.py)).unwrap_or(false));
        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "key_padding_mask").map(|x| x.is_none(ctx.py)).unwrap_or(false));
        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "use_separate_proj_weight")?.extract::<bool>(ctx.py)? == false);
        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "static_k").map(|x| x.is_none(ctx.py)).unwrap_or(false));
        assert!(py_node.getattr(ctx.py, "kwargs")?.get_item(ctx.py, "static_v").map(|x| x.is_none(ctx.py)).unwrap_or(false));

        let query_tensor_id = ctx.results[py_query_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let key_tensor_id = ctx.results[py_key_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let value_tensor_id = ctx.results[py_value_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let in_proj_weight_tensor_id = ctx.results[py_in_proj_weight_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let in_proj_bias_tensor_id = ctx.results[py_in_proj_bias_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let out_proj_weight_tensor_id = ctx.results[py_out_proj_weight_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let out_proj_bias_tensor_id = ctx.results[py_out_proj_bias_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor();
        let attn_mask_tensor_id = if !py_attn_mask_node.is_none(ctx.py) {
            Some(ctx.results[py_attn_mask_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tensor())
        } else {
            None
        };

        let query_tensor_shape = &ctx.graph[query_tensor_id].shape;
        let key_tensor_shape = &ctx.graph[key_tensor_id].shape;

        let L = query_tensor_shape[0];
        let N = query_tensor_shape[1];
        let E = query_tensor_shape[2];
        let S = key_tensor_shape[0];

        let has_attn_mask = !py_attn_mask_node.is_none(ctx.py);

        let node_id = RNodeId(ctx.graph.nodes.len());
        let tensor1_id = RTensorId(ctx.graph.tensors.len());
        let tensor2_id = tensor1_id + 1;

        let op = Rc::new(Op {
            py_name: "torch.nn.functional.multi_head_attention_forward".to_string(),
            codegen: Box::new(|py, graph, inputs| {
                let outputs = match inputs {
                    [query, key, value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias] => {
                        graph.call_method(py, "call_function", (py.eval("torch.nn.functional.multi_head_attention_forward", None, None)?, PyNone,
                            py_dict!(py, query => query, key => key, value => value, in_proj_weight => in_proj_weight, in_proj_bias => in_proj_bias, out_proj_weight => out_proj_weight, out_proj_bias => out_proj_bias)
                        ), None)?
                    },
                    [query, key, value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, attn_mask] => {
                        graph.call_method(py, "call_function", (py.eval("torch.nn.functional.multi_head_attention_forward", None, None)?, PyNone,
                            py_dict!(py, query => query, key => key, value => value, in_proj_weight => in_proj_weight, in_proj_bias => in_proj_bias, out_proj_weight => out_proj_weight, out_proj_bias => out_proj_bias, attn_mask => attn_mask)
                        ), None)?
                    },
                    _ => unreachable!()
                };
                Ok(smallvec![outputs.get_item(py, 0)?, outputs.get_item(py, 1)?])
            }),
            flops: Box::new(|shapes| {
                if let [query_shape, key_shape, ..] = shapes {
                    let L = &query_shape[0];
                    let N = &query_shape[1];
                    let E = &query_shape[2];
                    let S = &key_shape[0];

                    3. * L.clone() * N.clone() * E.clone() * S.clone() + 5. * L.clone() * S.clone() * N.clone() + 3. * L.clone() * N.clone() * E.clone() * S.clone()
                } else {
                    unreachable!()
                }
            }),
            info: [("has_attn_mask".to_string(), has_attn_mask.to_string())].into_iter().collect(),
        });

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            segment_id: *ctx.current_segment,
            shape: smallvec![L, N, E],
            communicatable: true
        });

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            segment_id: *ctx.current_segment,
            shape: smallvec![N, L, S],
            communicatable: true
        });

        let mut input_node_ids = smallvec![query_tensor_id, key_tensor_id, value_tensor_id, in_proj_weight_tensor_id, in_proj_bias_tensor_id, out_proj_weight_tensor_id, out_proj_bias_tensor_id];
        if has_attn_mask {
            input_node_ids.push(attn_mask_tensor_id.unwrap());
        }

        ctx.graph.nodes.push(RNode {
            inputs: input_node_ids,
            outputs: smallvec![tensor1_id, tensor2_id],
            instruction: RInstruction::Op(op)
        });

        ctx.graph[query_tensor_id].consumers.push(node_id);
        ctx.graph[key_tensor_id].consumers.push(node_id);
        ctx.graph[value_tensor_id].consumers.push(node_id);
        ctx.graph[in_proj_weight_tensor_id].consumers.push(node_id);
        ctx.graph[in_proj_bias_tensor_id].consumers.push(node_id);
        ctx.graph[out_proj_weight_tensor_id].consumers.push(node_id);
        ctx.graph[out_proj_bias_tensor_id].consumers.push(node_id);
        if has_attn_mask {
            ctx.graph[attn_mask_tensor_id.unwrap()].consumers.push(node_id);
        }

        ctx.results[py_id] = Some(EvalResult::Tuple(smallvec![tensor1_id, tensor2_id]));

        Ok(())
    }

    parsing_handlers.insert(py.eval("operator.getitem", None, None)?.as_ptr() as _, &handle_getitem);
    fn handle_getitem(ctx: ParserContext, py_node: PyObject) -> PyResult<()> {
        let py_id: usize = py_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract(ctx.py)?;

        let py_input_node = py_node.getattr(ctx.py, "args")?.get_item(ctx.py, 0)?;
        let n = py_node.getattr(ctx.py, "args")?.get_item(ctx.py, 1)?.extract::<usize>(ctx.py)?;

        let input_tuple = ctx.results[py_input_node.getattr(ctx.py, "meta")?.get_item(ctx.py, "id")?.extract::<usize>(ctx.py)?].as_ref().unwrap().as_tuple();

        ctx.results[py_id] = Some(EvalResult::Tensor(input_tuple[n]));

        Ok(())
    }


    Ok(parsing_handlers)
}

fn load_fx_graph(py: Python, py_graph_module: PyObject, py_input_shape_dict: PyObject) -> PyResult<RGraph> {
    let mut graph = RGraph::default();

    let parsing_handlers = initialize_parsing_handlers(py)?;

    let n_nodes = py_graph_module.getattr(py, "graph")?.getattr(py, "nodes")?.len(py)?;

    let mut results: Vec<Option<EvalResult>> = vec![None; n_nodes];
    let mut current_segment = SegmentId(0);

    for py_node in py_graph_module.getattr(py, "graph")?.getattr(py, "nodes")?.iter(py)? {
        let py_node = py_node?;
        let op_str: String = py_node.getattr(py, "op")?.extract(py)?;
        let py_id: usize = py_node.getattr(py, "meta")?.get_item(py, "id")?.extract(py)?;

        // memo when adding a node:
        // if the node has input, link the consumer of the inputs
        // if the node has output, set the result

        match &op_str[..] {
            "placeholder" => {
                let name: String = py_node.getattr(py, "target")?.extract(py)?;
                let shape: Vec<usize> = py_input_shape_dict.get_item(py, &name)?.extract(py)?;

                let node_id = RNodeId(graph.nodes.len());
                let tensor_id = RTensorId(graph.tensors.len());

                graph.nodes.push(RNode {
                    inputs: smallvec![],
                    outputs: smallvec![tensor_id],
                    instruction: RInstruction::Placeholder(name.clone()),
                });

                graph.tensors.push(RTensor {
                    producer: node_id,
                    consumers: smallvec![],
                    segment_id: current_segment,
                    shape: shape.into(),
                    communicatable: false
                });

                results[py_id] = Some(EvalResult::Tensor(tensor_id));
            },

            "get_attr" => {
                let name: String = py_node.getattr(py, "target")?.extract(py)?;

                let shape: Vec<usize> = py.eval(
                    "get_shape_of_param_or_buffer(graph_module, node)",
                    None, Some(&py_dict!(py, graph_module => py_graph_module, node => py_node))
                )?.extract(py)?;

                let node_id = RNodeId(graph.nodes.len());
                let tensor_id = RTensorId(graph.tensors.len());

                graph.nodes.push(RNode {
                    inputs: smallvec![],
                    outputs: smallvec![tensor_id],
                    instruction: RInstruction::GetAttr(name),
                });

                graph.tensors.push(RTensor {
                    producer: node_id,
                    consumers: smallvec![],
                    segment_id: current_segment,
                    shape: shape.into(),
                    communicatable: false
                });

                results[py_id] = Some(EvalResult::Tensor(tensor_id));
            },

            "call_function" => {
                let ctx = ParserContext {
                    py,
                    graph: &mut graph,
                    current_segment: &mut current_segment,
                    results: &mut results
                };

                eprintln!("target_function: {:?}", py_node.getattr(py, "target"));

                parsing_handlers[&(py_node.getattr(py, "target")?.as_ptr() as _)](ctx, py_node)?;
            },

            "call_method" => {
                let ctx = ParserContext {
                    py,
                    graph: &mut graph,
                    current_segment: &mut current_segment,
                    results: &mut results
                };

                let tensor_class = py.eval("torch.Tensor", None, None)?;
                let target_method = py_node.getattr(py, "target")?;

                parsing_handlers[&(tensor_class.getattr(ctx.py, target_method)?.as_ptr() as _)](ctx, py_node)?;
            }

            "output" => {
                if graph.nodes.iter().any(|node| matches!(node.instruction, RInstruction::Output)) {
                    panic!("Multiple outputs in the graph");
                }

                let node_id = RNodeId(graph.nodes.len());

                let py_input_node = py_node.getattr(py, "args")?.get_item(py, 0)?;
                let py_input_id: usize = py_input_node.getattr(py, "meta")?.get_item(py, "id")?.extract(py)?;

                let input_tensor_id = match results[py_input_id].as_ref().unwrap() {
                    EvalResult::Tensor(tensor_id) => *tensor_id,
                    _ => unreachable!()
                };

                graph[input_tensor_id].consumers.push(node_id);

                graph.nodes.push(RNode {
                    inputs: smallvec![input_tensor_id],
                    outputs: smallvec![],
                    instruction: RInstruction::Output,
                });
            },

            _ => unreachable!()
        }
    }

    graph.n_segments = current_segment.0 + 1;

    Ok(graph)
}

fn analyze_rgraph(rgraph: &RGraph) -> Vec<HoareTriple> {
    let mut triples = vec![];

    let mut add_triple = |pre_conditions, post_conditions, instruction, codegen, profile| {
        triples.push(HoareTriple {
            pre_conditions,
            post_conditions,
            negative_post_conditions: vec![],
            instruction,
            codegen,
            profile
        });
    };

    // Placeholder, GetAttr, Output
    for node in rgraph.nodes.iter() {
        match &node.instruction {
            RInstruction::Placeholder(placeholder_name) => {
                let tensor_id = node.outputs[0];

                add_triple(
                    smallvec![],
                    smallvec![Property::identity(tensor_id)],
                    format!("placeholder_unsharded(\"{placeholder_name}\")"),
                    Rc::new({
                        let placeholder_name = placeholder_name.clone();
                        move |ctx| {
                            let py_result = ctx.fx_placeholder(&placeholder_name)?;
                            ctx.set_property_implementation(Property::identity(tensor_id), py_result);
                            Ok(())
                        }
                    }),
                    Rc::new(|ctx| Default::default())
                );

                for (dim, &length) in rgraph[tensor_id].shape.iter().enumerate() {
                    let dim = dim as Dimension;

                    add_triple(
                        smallvec![],
                        smallvec![Property::gather(tensor_id, dim)],
                        format!("placeholder_shard(\"{placeholder_name}\", dim={dim}])"),
                        Rc::new({
                            let placeholder_name = placeholder_name.clone();
                            let segment_id = rgraph[tensor_id].segment_id;
                            move |ctx| {
                                let py_complete_placeholder = ctx.fx_placeholder(&placeholder_name)?;
                                let chunk_lengths = sharding_round(length, &ctx.sharding_ratios[segment_id.0]);
                                let py_chunks = ctx.fx_call_method("split", (py_complete_placeholder, chunk_lengths, dim), None)?;
                                let py_chunk = ctx.fx_call_function("operator.getitem", (py_chunks, ctx.rank), None)?;
                                ctx.set_property_implementation(Property::gather(tensor_id, dim), py_chunk);
                                Ok(())
                            }
                        }),
                        Rc::new(|ctx| { Default::default() })
                    );
                }
            }

            RInstruction::GetAttr(parameter_name) => {
                let tensor_id = node.outputs[0];

                add_triple(
                    smallvec![],
                    smallvec![Property::identity(tensor_id)],
                    format!("get_attr_unsharded(\"{parameter_name}\")"),
                    Rc::new({
                        let parameter_name = parameter_name.clone();
                        move |ctx| {
                            let py_parameter = ctx.fx_get_attr(&parameter_name)?;
                            let py_replicated = ctx.fx_call_function("collectives.replicate", (py_parameter,), None)?;
                            ctx.set_property_implementation(Property::identity(tensor_id), py_replicated);
                            Ok(())
                        }
                    }),
                    Rc::new({
                        move |ctx| {
                            let size = ctx.get_shape_by_property(Property::identity(tensor_id)).iter().cloned().product::<Expression>();
                            let forward_profile = Default::default();
                            let backward_profile = Profile { all_reduce: size, ..Default::default() };
                            (forward_profile, backward_profile)
                        }
                    })
                );

                for (dim, &length) in rgraph[tensor_id].shape.iter().enumerate() {
                    let dim = dim as Dimension;

                    add_triple(
                        smallvec![],
                        smallvec![Property::gather(tensor_id, dim)],
                        format!("get_attr_shard(\"{parameter_name}\", dim={dim}])"),
                        Rc::new({
                            let parameter_name = parameter_name.clone();
                            let segment_id = rgraph[tensor_id].segment_id;
                            move |ctx| {
                                let py_parameter = ctx.fx_get_attr(&parameter_name)?;
                                ctx.shard_inplace(&parameter_name, &sharding_round(length, &ctx.sharding_ratios[segment_id.0]), dim)?;
                                ctx.set_property_implementation(Property::gather(tensor_id, dim), py_parameter);
                                Ok(())
                            }
                        }),
                        Rc::new(|ctx| { Default::default() })
                    );
                }
            }

            RInstruction::Output => {
                let tensor_id = node.inputs[0];

                add_triple(
                    smallvec![Property::reduce(tensor_id)],
                    smallvec![Property::Finished],
                    format!("output"),
                    Rc::new(move |ctx| {
                        let py_input = ctx.get_property_implementation(Property::reduce(tensor_id));
                        ctx.fx_output(py_input)?;
                        Ok(())
                    }),
                    Rc::new(|ctx| { Default::default() })
                );
            }

            _ => {}
        }
    }

    // communication & dynamic slice
    for (tensor_id, tensor) in rgraph.tensors.iter().enumerate() {
        let tensor_id = RTensorId(tensor_id);

        if !tensor.communicatable {
            continue;
        }

        for dim in 0..tensor.n_dims() {
            add_triple(
                smallvec![Property::gather(tensor_id, dim)],
                smallvec![Property::identity(tensor_id)],
                format!("all_gather(dim={dim})"),
                Rc::new({
                    let segment_id = tensor.segment_id;
                    move |ctx| {
                        let py_input = ctx.get_property_implementation(Property::gather(tensor_id, dim));
                        let py_result = ctx.fx_call_function("collectives.all_gather", (py_input, dim, sharding_round(ctx.get_shape_by_property(Property::identity(tensor_id))[dim as usize], &ctx.sharding_ratios[segment_id.0]), ctx.rank), None)?;
                        ctx.set_property_implementation(Property::identity(tensor_id), py_result);
                        Ok(())
                    }
                }),
                Rc::new(move |ctx| {
                    let size = ctx.get_shape_by_property(Property::gather(tensor_id, dim)).iter().cloned().product::<Expression>();
                    let forward_profile = Profile { all_gather: size.clone() * ctx.profiler.cluster_info.n_devices() as f64, ..Default::default() };
                    let backward_profile = Profile { reduce_scatter: size.clone() * ctx.profiler.cluster_info.n_devices() as f64, ..Default::default() };
                    (forward_profile, backward_profile)
                })
            );

            add_triple(
                smallvec![Property::identity(tensor_id)],
                smallvec![Property::gather(tensor_id, dim)],
                format!("dynamic_slice(dim={dim})"),
                Rc::new({
                    let segment_id = tensor.segment_id;
                    move |ctx| {
                        let py_input = ctx.get_property_implementation(Property::identity(tensor_id));
                        let py_result = ctx.fx_call_function("collectives.dynamic_slice", (py_input, dim, sharding_round(ctx.get_shape_by_property(Property::identity(tensor_id))[dim as usize], &ctx.sharding_ratios[segment_id.0]), ctx.rank), None)?;
                        ctx.set_property_implementation(Property::gather(tensor_id, dim), py_result);
                        Ok(())
                    }
                }),
                Rc::new(move |ctx| { Default::default() })
            );

            add_triple(
                smallvec![Property::reduce(tensor_id)],
                smallvec![Property::gather(tensor_id, dim)],
                format!("reduce_scatter(dim={dim})"),
                Rc::new({
                    let segment_id = tensor.segment_id;
                    move |ctx| {
                        let py_input = ctx.get_property_implementation(Property::reduce(tensor_id));
                        let py_result = ctx.fx_call_function("collectives.reduce_scatter", (py_input, dim, sharding_round(ctx.get_shape_by_property(Property::identity(tensor_id))[dim as usize], &ctx.sharding_ratios[segment_id.0]), ctx.rank), None)?;
                        ctx.set_property_implementation(Property::gather(tensor_id, dim), py_result);
                        Ok(())
                    }
                }),
                Rc::new(move |ctx| {
                    let size = ctx.get_shape_by_property(Property::gather(tensor_id, dim)).iter().cloned().product::<Expression>();
                    let forward_profile = Profile { reduce_scatter: size.clone() * ctx.profiler.cluster_info.n_devices() as f64, ..Default::default() };
                    let backward_profile = Profile { all_gather: size.clone() * ctx.profiler.cluster_info.n_devices() as f64, ..Default::default() };
                    (forward_profile, backward_profile)
                })
            );
        }

        add_triple(
            smallvec![Property::reduce(tensor_id)],
            smallvec![Property::identity(tensor_id)],
            format!("all_reduce"),
            Rc::new(move |ctx| {
                let py_input = ctx.get_property_implementation(Property::reduce(tensor_id));
                let py_result = ctx.fx_call_function("collectives.all_reduce", (py_input,), None)?;
                ctx.set_property_implementation(Property::identity(tensor_id), py_result);
                Ok(())
            }),
            Rc::new(move |ctx| {
                let size = ctx.get_shape_by_property(Property::identity(tensor_id)).iter().cloned().product::<Expression>();
                let forward_profile = Profile { all_reduce: size.clone(), ..Default::default() };
                let backward_profile = Profile { all_reduce: size.clone(), ..Default::default() };
                (forward_profile, backward_profile)
            })
        );

        for i in 0..tensor.n_dims() {
            for j in 0..tensor.n_dims() {
                if i != j {
                    add_triple(
                        smallvec![Property::gather(tensor_id, i)],
                        smallvec![Property::gather(tensor_id, j)],
                        format!("all_to_all(cat={i}, split={j})"),
                        Rc::new({
                            let segment_id = tensor.segment_id;
                            move |ctx| {
                                let py_input = ctx.get_property_implementation(Property::gather(tensor_id, i));
                                let py_result = ctx.fx_call_function("collectives.all_to_all", (
                                    py_input,
                                    j,
                                    i,
                                    sharding_round(ctx.get_shape_by_property(Property::identity(tensor_id))[j as usize], &ctx.sharding_ratios[segment_id.0]),
                                    sharding_round(ctx.get_shape_by_property(Property::identity(tensor_id))[i as usize], &ctx.sharding_ratios[segment_id.0]),
                                    ctx.rank
                                ), None)?;
                                ctx.set_property_implementation(Property::gather(tensor_id, j), py_result);
                                Ok(())
                            }
                        }),
                        Rc::new(move |ctx| {
                            let size = ctx.get_shape_by_property(Property::gather(tensor_id, i)).iter().cloned().product::<Expression>();
                            let forward_profile = Profile { all_to_all: size.clone() * ctx.profiler.cluster_info.n_devices() as f64, ..Default::default() };
                            let backward_profile = Profile { all_to_all: size.clone() * ctx.profiler.cluster_info.n_devices() as f64, ..Default::default() };
                            (forward_profile, backward_profile)
                        })
                    );
                }
            }
        }
    }

    macro_rules! for_each_op {
        ($op_name: expr, |$node: ident, $op: ident| $body: block) => {{
            for $node in rgraph.nodes.iter() {
                if let RInstruction::Op($op) = &$node.instruction && $op.py_name == $op_name {
                    $body
                }
            }
        }}
    }

    let mut add_comp_triple = |pre_conditions: SVec<Property, 4>, post_conditions: SVec<Property>, op: Rc<Op>| {
        add_triple(
            pre_conditions.clone(),
            post_conditions.clone(),
            op.py_name.clone(),
            Rc::new({
                let op = op.clone();
                let pre_conditions = pre_conditions.clone();
                move |ctx| {
                    let inputs: Vec<_> = pre_conditions.iter().map(|p| ctx.get_property_implementation(*p)).collect();
                    let outputs = (op.codegen)(ctx.py, &ctx.graph, &inputs)?;
                    for (output_property, py_output) in post_conditions.iter().filter(|p| matches!(p, Property::HasTensor(_, _))).zip(outputs) {
                        ctx.set_property_implementation(*output_property, py_output);
                    }
                    Ok(())
                }
            }),
            Rc::new(move |ctx| {
                let shapes: Vec<_> = pre_conditions.iter().map(|p| ctx.get_shape_by_property(*p)).collect();
                let flops = (op.flops)(&shapes);
                let forward_profile = Profile { flops: flops.clone(), ..Default::default() };
                let backward_profile = Profile { flops: 2. * flops, ..Default::default() };
                (forward_profile, backward_profile)
            })
        )
    };

    // Linear
    for_each_op!("torch.nn.functional.linear", |node, op| {
        add_comp_triple(
            node.inputs.iter().cloned().map(Property::identity).collect(),
            node.outputs.iter().cloned().map(Property::identity).collect(),
            op.clone(),
        );

        // data parallelism
        for dim in 0..rgraph[node.inputs[0]].n_dims() - 1 {
            add_comp_triple(
                smallvec![
                    Property::gather(node.inputs[0], dim),
                    Property::identity(node.inputs[1]),
                    Property::identity(node.inputs[2]),
                ],
                smallvec![Property::gather(node.outputs[0], dim)],
                op.clone(),
            );
        }

        // feature partition
        add_comp_triple(
            smallvec![
                Property::identity(node.inputs[0]),
                Property::gather(node.inputs[1], 0),
                Property::gather(node.inputs[2], 0),
            ],
            smallvec![Property::gather(node.outputs[0], rgraph[node.outputs[0]].n_dims() - 1)],
            op.clone(),
        );

        // reduction?
        // this requires arithemetic replacement (change to matmul + allreduce + add)
        // we also hit Rust aliasing rule here as the loop already borrows the graph
    });

    // Sigmoid
    for_each_op!("torch.sigmoid", |node, op| {
        add_comp_triple(
            node.inputs.iter().cloned().map(Property::identity).collect(),
            node.outputs.iter().cloned().map(Property::identity).collect(),
            op.clone(),
        );

        for dim in 0..rgraph[node.inputs[0]].n_dims() {
            add_comp_triple(
                smallvec![Property::gather(node.inputs[0], dim)],
                smallvec![Property::gather(node.outputs[0], dim)],
                op.clone(),
            );
        }
    });

    // Sum
    for_each_op!("torch.sum", |node, op| {
        add_comp_triple(
            node.inputs.iter().cloned().map(Property::identity).collect(),
            node.outputs.iter().cloned().map(Property::identity).collect(),
            op.clone(),
        );

        for dim in 0..rgraph[node.inputs[0]].n_dims() {
            add_comp_triple(
                smallvec![Property::gather(node.inputs[0], dim)],
                smallvec![Property::reduce(node.outputs[0])],
                op.clone(),
            );
        }

        add_comp_triple(
            smallvec![Property::reduce(node.inputs[0])],
            smallvec![Property::reduce(node.outputs[0])],
            op.clone(),
        );
    });

    // transpose
    for_each_op!("torch.transpose", |node, op| {
        eprintln!("here");

        add_comp_triple(
            smallvec![Property::identity(node.inputs[0])],
            smallvec![Property::identity(node.outputs[0])],
            op.clone(),
        );

        add_comp_triple(
            smallvec![Property::reduce(node.inputs[0])],
            smallvec![Property::reduce(node.outputs[0])],
            op.clone(),
        );

        let dim0: Dimension = op.info["dim0"].parse().unwrap();
        let dim1: Dimension = op.info["dim1"].parse().unwrap();

        for dim in 0..rgraph[node.inputs[0]].n_dims() {
            if dim == dim0 {
                add_comp_triple(
                    smallvec![Property::gather(node.inputs[0], dim0)],
                    smallvec![Property::gather(node.outputs[0], dim1)],
                    op.clone(),
                );
            } else if dim == dim1 {
                add_comp_triple(
                    smallvec![Property::gather(node.inputs[0], dim1)],
                    smallvec![Property::gather(node.outputs[0], dim0)],
                    op.clone(),
                );
            } else {
                add_comp_triple(
                    smallvec![Property::gather(node.inputs[0], dim)],
                    smallvec![Property::gather(node.outputs[0], dim)],
                    op.clone(),
                );
            }
        }
    });

    // attention
    for_each_op!("torch.nn.functional.multi_head_attention_forward", |node, op| {
        let _has_attn_mask: bool = op.info["has_attn_mask"].parse().unwrap();

        add_comp_triple(
            node.inputs.iter().map(|x| Property::identity(*x)).collect(),
            node.outputs.iter().map(|x| Property::identity(*x)).collect(),
            op.clone(),
        );

        let mut inputs = smallvec![
            Property::gather(node.inputs[0], 1), // query
            Property::gather(node.inputs[1], 1), // key
            Property::gather(node.inputs[2], 1), // value
        ];

        for x in 3..node.inputs.len() {
            inputs.push(Property::identity(node.inputs[x]));
        }

        add_comp_triple(
            inputs,
            smallvec![Property::gather(node.outputs[0], 1), Property::gather(node.outputs[0], 0)],
            op.clone(),
        );

        let mut inputs = smallvec![
            Property::gather(node.inputs[0], 2), // query
            Property::gather(node.inputs[1], 2), // key
            Property::gather(node.inputs[2], 2), // value
            Property::gather(node.inputs[3], 0), // in_proj_weight
            Property::gather(node.inputs[4], 0), // in_proj_bias
            Property::gather(node.inputs[5], 1), // out_proj_weight
        ];

        for x in 6..node.inputs.len() {
            inputs.push(Property::identity(node.inputs[x]));
        }

        add_comp_triple(
            inputs,
            smallvec![Property::reduce(node.outputs[0]), Property::reduce(node.outputs[0])],
            op.clone(),
        )
    });

    triples
}

mod heuristics {
    use super::*;

    fn get_rtensor_ids_from_conditions(conditions: &[Property]) -> Vec<RTensorId> {
        conditions.iter()
            .flat_map(|p| match *p {
                Property::HasTensor(tensor_id, _) => Some(tensor_id),
                _ => None
            }).collect()
    }

    /// only allow up to one communication per rtensor
    pub fn unique_communication(triples: &mut Vec<HoareTriple>, default_properties: &mut Vec<Property>) {
        for triple in triples {
            let input_tensor_ids = get_rtensor_ids_from_conditions(&triple.pre_conditions);
            let output_tensor_ids = get_rtensor_ids_from_conditions(&triple.post_conditions);

            if input_tensor_ids.len() == 1 && output_tensor_ids.len() == 1 && input_tensor_ids[0] == output_tensor_ids[0] {
                triple.pre_conditions.push(Property::AllowCommunication(input_tensor_ids[0]));
                triple.negative_post_conditions.push(Property::AllowCommunication(input_tensor_ids[0]));
                default_properties.push(Property::AllowCommunication(input_tensor_ids[0]));
            }
        }
    }

    pub fn unique_computation(triples: &mut Vec<HoareTriple>, default_properties: &mut Vec<Property>) {
        for triple in triples {
            let input_tensor_ids = get_rtensor_ids_from_conditions(&triple.pre_conditions);
            let output_tensor_ids = get_rtensor_ids_from_conditions(&triple.post_conditions);

            if output_tensor_ids.iter().all(|tensor_id| !input_tensor_ids.contains(tensor_id)) {
                for tensor_id in output_tensor_ids {
                    triple.pre_conditions.push(Property::AllowComputation(tensor_id));
                    triple.negative_post_conditions.push(Property::AllowComputation(tensor_id));
                    default_properties.push(Property::AllowComputation(tensor_id));
                }
            }
        }
    }

    /// fuse free triples into its consumers. Free triples are those have no tensor input (Placeholder and GetAttr)
    pub fn fuse_free_triple(triples: &mut Vec<HoareTriple>, _default_properties: &mut Vec<Property>) {
        let mut i = 0;
        while i < triples.len() {
            let input_tensor_ids = get_rtensor_ids_from_conditions(&triples[i].pre_conditions);
            let output_tensor_ids = get_rtensor_ids_from_conditions(&triples[i].post_conditions);

            if input_tensor_ids.is_empty() && output_tensor_ids.len() == 1 { // free triple
                let free_triple = triples.remove(i); // idea: swap_remove for performance?
                let output_property = *free_triple.post_conditions.iter().find(|x| matches!(x, Property::HasTensor(_, _))).unwrap();

                // idea: can make index here if the number of triples is huge

                let mut j = 0;
                while j < triples.len() {
                    let triple = &triples[j];
                    if triple.pre_conditions.contains(&output_property) && free_triple.negative_post_conditions.iter().all(|x| !triple.pre_conditions.contains(x)) {
                        triples.push(free_triple.fuse_into(triple));
                    }
                    j += 1;
                }
            } else {
                i += 1
            }
        }
    }

    pub fn fuse_communication(triples: &mut Vec<HoareTriple>, _default_properties: &mut Vec<Property>) {
        let mut i = 0;
        while i < triples.len() {
            let input_tensor_ids = get_rtensor_ids_from_conditions(&triples[i].pre_conditions);
            let output_tensor_ids = get_rtensor_ids_from_conditions(&triples[i].post_conditions);

            if input_tensor_ids.len() == 1 && output_tensor_ids.len() == 1 && input_tensor_ids[0] == output_tensor_ids[0] { // communication
                let comm_triple = triples.remove(i);
                let output_property = *comm_triple.post_conditions.iter().find(|x| matches!(x, Property::HasTensor(_, _))).unwrap();

                let mut j = 0;
                while j < triples.len() {
                    let triple = &triples[j];
                    if triple.pre_conditions.contains(&output_property) && comm_triple.negative_post_conditions.iter().all(|x| !triple.pre_conditions.contains(x)) {
                        triples.push(comm_triple.fuse_into(triple));
                    }
                    j += 1;
                }
            } else {
                i += 1
            }
        }
    }
}

#[derive(Debug)]
struct ClusterInfo {
    device_flops: Vec<f64>,
    all_reduce_bandwidth: f64,
    all_gather_bandwidth: f64,
    all_to_all_bandwidth: f64,
    reduce_scatter_bandwidth: f64,
}

impl ClusterInfo {
    fn n_devices(&self) -> usize {
        self.device_flops.len()
    }
}

// idea: if calculating it is time consuming, make a struct called sharding plan wraps the sharding ratios and caches the sharding_round results for each length
fn sharding_round(full_length: usize, sharding_ratios: &[f64]) -> Vec<usize> {
    let mut sharding_lengths: Vec<_> = sharding_ratios.iter().map(|x| (x * full_length as f64) as usize).collect();
    assert!(sharding_lengths.iter().sum::<usize>() <= full_length);
    while sharding_lengths.iter().sum::<usize>() < full_length {
        let max_diff_index = sharding_ratios.iter()
            .zip(sharding_lengths.iter())
            .enumerate()
            .max_by_key(|(_, (&ratio, &length))| FloatOrd(ratio - length as f64 / full_length as f64))
            .unwrap().0;

        sharding_lengths[max_diff_index] += 1;
    }
    sharding_lengths
}

fn sharding_symbolic(full_length: usize, sharding_ratios: &[Expression]) -> Vec<Expression> {
    sharding_ratios.iter().map(|x| x.clone() * (full_length as f64)).collect()
}

new_usize_type!(pub, SymbolId);

#[derive(Debug, Clone)]
enum Expression {
    Symbol(SymbolId, f64),
    Constant(f64),
    Linear(Vec<f64>, f64)
}

impl Expression {
    fn symbol(symbol_id: SymbolId) -> Self {
        Expression::Symbol(symbol_id, 1.)
    }

    fn constant(value: f64) -> Self {
        Expression::Constant(value)
    }

    fn to_linear(self) -> Self {
        match self {
            Expression::Symbol(symbol_id, coefficient) => {
                let mut coefficients = vec![0.; symbol_id.0 + 1];
                coefficients[symbol_id.0] = coefficient;
                Expression::Linear(coefficients, 0.)
            }
            Expression::Constant(value) => Expression::Linear(vec![], value),
            Expression::Linear(_, _) => self
        }
    }

    fn instantialize(&self, symbol_values: &[f64]) -> f64 {
        match self {
            Expression::Symbol(symbol_id, coefficient) => symbol_values[symbol_id.0] * coefficient,
            Expression::Constant(value) => *value,
            Expression::Linear(coefficients, constant) => {
                let mut value = *constant;
                for (i, coefficient) in coefficients.iter().enumerate() {
                    value += symbol_values[i] * coefficient;
                }
                value
            }
        }
    }

    fn unwrap_constant(&self) -> f64 {
        match self {
            Expression::Constant(value) => *value,
            _ => panic!("Expression is not a constant")
        }
    }
}

impl Add for Expression {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // fallback implementation
            (Expression::Linear(lhs_coefficients, lhs_constant), Expression::Linear(rhs_coefficients, rhs_constant)) => {
                let mut coefficients = vec![0.; lhs_coefficients.len().max(rhs_coefficients.len())];
                for (i, coefficient) in lhs_coefficients.into_iter().enumerate() {
                    coefficients[i] += coefficient;
                }
                for (i, coefficient) in rhs_coefficients.into_iter().enumerate() {
                    coefficients[i] += coefficient;
                }
                Expression::Linear(coefficients, lhs_constant + rhs_constant)
            }

            // specialized implementation
            (Expression::Symbol(lhs_symbol_id, lhs_coefficient), Expression::Symbol(rhs_symbol_id, rhs_coefficient)) if lhs_symbol_id == rhs_symbol_id => {
                Expression::Symbol(lhs_symbol_id, lhs_coefficient + rhs_coefficient)
            }
            (Expression::Constant(lhs_constant), Expression::Constant(rhs_constant)) => {
                Expression::Constant(lhs_constant + rhs_constant)
            }

            // catch-all
            (lhs @ _, rhs @ _) => lhs.to_linear() + rhs.to_linear()
        }
    }
}

impl Add<f64> for Expression {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        self + Expression::constant(rhs)
    }
}

impl Add<Expression> for f64 {
    type Output = Expression;

    fn add(self, rhs: Expression) -> Self::Output {
        Expression::constant(self) + rhs
    }
}

impl Mul for Expression {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Expression::Constant(lhs_constant), Expression::Constant(rhs_constant)) => {
                Expression::Constant(lhs_constant * rhs_constant)
            }
            (Expression::Constant(constant), Expression::Symbol(symbol_id, coefficient)) => {
                Expression::Symbol(symbol_id, coefficient * constant)
            }
            (Expression::Constant(lhs_constant), Expression::Linear(coefficients, rhs_constant)) => {
                Expression::Linear(coefficients.into_iter().map(|x| x * lhs_constant).collect(), rhs_constant * lhs_constant)
            }

            (lhs @ _, rhs @ Expression::Constant(_)) => rhs * lhs,

            x @ _ => panic!("quadratic expression: {:?}", x)
        }
    }
}

impl Mul<f64> for Expression {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self * Expression::constant(rhs)
    }
}

impl Mul<Expression> for f64 {
    type Output = Expression;

    fn mul(self, rhs: Expression) -> Self::Output {
        Expression::constant(self) * rhs
    }
}

impl Product for Expression {
    fn product<I: Iterator<Item=Self>>(iter: I) -> Self {
        iter.fold(Expression::constant(1.), |acc, x| acc * x)
    }
}

impl Div<f64> for Expression {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        self * (1. / rhs)
    }
}

impl Default for Expression {
    fn default() -> Self {
        Expression::Constant(0.)
    }
}

impl<T: Into<f64>> From<T> for Expression {
    fn from(value: T) -> Self {
        Expression::Constant(value.into())
    }
}

fn sharding_ratio_optimization(program: &Program, triple_set: &IndexedHoareTripleSet, sharding_ratios: &[Vec<SymbolId>], profiler: &Profiler) {
    let mut model = coin_cbc::Model::default();

    model.set_parameter("slogLevel", "0");
    model.set_parameter("logLevel", "0");
    model.add_integer(); // CBC's bug: the parameters are not passed to the solver if the problem is pure LP

    model.set_obj_sense(coin_cbc::Sense::Minimize);

    let n_segments = profiler.rgraph.n_segments;
    let n_devices = profiler.cluster_info.n_devices();
    let n_stages = program.triple_ids.len();

    // let mut sharding_ratios_obj_coeff = vec![0.; n_segments * n_devices];
    let mut sharding_ratios_cbc: Vec<_> = (0..n_segments * n_devices).into_iter().map(|_| {
        let x = model.add_col();
        // model.set_continuous(x);
        model.set_col_lower(x, 0.);
        model.set_col_upper(x, 1.);
        model.set_obj_coeff(x, 0.);
        x
    }).collect();

    for triple_id in program.triple_ids.iter() {
        let triple = &triple_set.triples[triple_id.0];
        let (computation_times, communication_times) = triple.get_cost_symbolic(profiler, &sharding_ratios.iter().map(|s| {
            s.iter().map(|d| Expression::symbol(*d)).collect()
        }).collect::<Vec<_>>());

        let computation_max = model.add_col();
        model.set_col_lower(computation_max, 0.);
        model.set_col_upper(computation_max, 1.);
        model.set_obj_coeff(computation_max, 1.);
        for computation_time in computation_times {
            let row = model.add_row();
            model.set_row_upper(row, 0.);
            model.set_weight(row, computation_max, -1.);

            if let Expression::Linear(coefficients, constant) = computation_time.to_linear() {
                for (i, coeff) in coefficients.into_iter().enumerate() {
                    model.set_weight(row, sharding_ratios_cbc[i], coeff);
                }
            } else {
                unreachable!()
            }
        }

        let communication_max = model.add_col();
        model.set_col_lower(communication_max, 0.);
        model.set_col_upper(communication_max, 1.);
        model.set_obj_coeff(communication_max, 1.);
        for communication_time in communication_times {
            let row = model.add_row();
            model.set_row_upper(row, 0.);
            model.set_weight(row, communication_max, -1.);

            if let Expression::Linear(coefficients, constant) = communication_time.to_linear() {
                for (i, coeff) in coefficients.into_iter().enumerate() {
                    model.set_weight(row, sharding_ratios_cbc[i], coeff);
                }
            } else {
                unreachable!()
            }
        }

    }

    for s in 0..n_segments {
        let row = model.add_row();
        model.set_row_lower(row, 1.);
        model.set_row_upper(row, 1.);
        for d in 0..n_devices {
            model.set_weight(row, sharding_ratios_cbc[sharding_ratios[s][d].0], 1.);
        }
    }

    // model.to_raw().write_mps(&std::ffi::CString::new("sharding_ratio").unwrap());

    let sol = model.solve();
    eprintln!("=== sharding ratios ===");
    for i in 0..n_segments {
        eprint!("[");
        for j in 0..n_devices {
            if j > 0 {
                eprint!(" ");
            }
            eprint!("{}", sol.col(sharding_ratios_cbc[sharding_ratios[i][j].0]));
        }
        eprintln!("]");
    }
}
