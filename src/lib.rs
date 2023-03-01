#![allow(unused)]
#![allow(non_upper_case_globals)]
#![feature(let_chains)]

use std::ops::{Index, IndexMut};
use std::sync::{atomic::AtomicBool, Arc};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt::{Display, Debug, Formatter};
use std::cmp::Ordering;
use float_ord::FloatOrd;
use cpython::{PyResult, PyTuple, ToPyObject, PythonObject, ObjectProtocol, Python, PyList, PyObject, PyDict, PyClone};
use smallvec::{SmallVec, smallvec};

// todo: benchmark change it to 4?
pub type SVec<T, const N: usize = 3> = SmallVec<[T; N]>;

static CTRLC_TRAPPED: AtomicBool = AtomicBool::new(false);
static CTRLC_RECEIVED: AtomicBool = AtomicBool::new(false);

cpython::py_module_initializer!(hetspmd, |py, m| {
    if !CTRLC_TRAPPED.load(std::sync::atomic::Ordering::Relaxed) {
        ctrlc::set_handler(|| {
            CTRLC_RECEIVED.store(true, std::sync::atomic::Ordering::Relaxed)
        }).unwrap();
    }

    #[allow(clippy::manual_strip)]
    m.add(py, "main", cpython::py_fn!(py, main(py_graph_module: PyObject, py_config: PyDict) -> PyResult<PyList> {
        let py_input_shape_dict = py_config.get_item(py, "input_shape").unwrap();
        let (rgraph, module_info) = load_fx_graph(py, py_graph_module.clone_ref(py), py_input_shape_dict)?;

        eprintln!("graph: {rgraph:#?}");
        eprintln!("module_info: {module_info:#?}");

        let mut triples = analyze_rgraph(&rgraph, &module_info);

        let mut default_properties = vec![];
        default_properties.extend(heuristics::compute_only_once(&mut triples, &rgraph));
        default_properties.extend(heuristics::ordered_communication(&mut triples, &rgraph));
        default_properties.extend(heuristics::ordered_placeholder(&mut triples, &rgraph));
        default_properties.extend(heuristics::ordered_get_attr(&mut triples, &rgraph));

        // for triple in triples.iter() {
        //     eprintln!("{triple}");
        // }

        let cluster_info = ClusterInfo {
            n_devices: 4,
            device_flops: vec![4139214925014.; 4],
            all_reduce_bandwidth: 611692856.,
            all_gather_bandwidth: 1224592728.,
            reduce_scatter_bandwidth: 1130230706.,
            all_to_all_bandwidth: 10701240728.
        };

        let profiler = Profiler {
            rgraph: &rgraph,
            module_info: &module_info,
            cluster_info: &cluster_info
        };

        a_star(&triples, &default_properties, &profiler);
        Ok(PyList::new(py, &[]))
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
    }
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

new_usize_type!(pub, OpId);
new_usize_type!(pub, ParameterId);
new_usize_type!(pub, PlaceholderId);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Collective {
    AllGather(u8),
    AllReduce,
    ReduceScatter(u8),
    AllToAll(u8, u8), // split_dim, cat_dim
    Replicate,
    DynamicSlice(u8)
}

impl Collective {
    fn conjugate(self) -> Option<Self> {
        match self {
            Collective::AllGather(dim) => Some(Collective::ReduceScatter(dim)),
            Collective::AllReduce => Some(Collective::AllReduce),
            Collective::ReduceScatter(dim) => Some(Collective::AllGather(dim)),
            Collective::AllToAll(split_dim, cat_dim) => Some(Collective::AllToAll(cat_dim, split_dim)),
            Collective::Replicate => Some(Collective::AllReduce),
            Collective::DynamicSlice(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HoareTriple {
    pre_conditions: SVec<Property>,
    post_conditions: SVec<Property, 1>,
    negative_post_conditions: Vec<Property>,
    instruction: DInstruction,
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
    AllowPlaceholder(PlaceholderId),
    AllowGetAttr(ParameterId),
    AllowComputation(OpId), // or RNodeId?
}

impl Display for Property {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Property::HasTensor(tensor_id, relation) => {
                write!(f, "{}|{:?}", tensor_id.0, relation)
            }
            Property::Finished => write!(f, "finished"),
            Property::AllowCommunication(tensor_id) => {
                write!(f, "{}|allow_communication", tensor_id.0)
            }
            Property::AllowPlaceholder(placeholder_id) => {
                write!(f, "{}|allow_placeholder", placeholder_id.0)
            }
            Property::AllowGetAttr(parameter_id) => {
                write!(f, "{}|allow_get_attr", parameter_id.0)
            }
            Property::AllowComputation(op_id) => {
                write!(f, "{}|allow_computation", op_id.0)
            }
        }
    }
}

impl Property {
    fn identity(tensor_id: RTensorId) -> Property {
        Property::HasTensor(tensor_id, TensorRelation::Identity)
    }

    fn gather(tensor_id: RTensorId, dim: u8) -> Property {
        Property::HasTensor(tensor_id, TensorRelation::Gather(dim))
    }

    fn reduce(tensor_id: RTensorId) -> Property {
        Property::HasTensor(tensor_id, TensorRelation::Reduce)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorRelation {
    Gather(u8),
    Reduce,
    Identity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ShardingForm {
    Sharded(u8),
    Unsharded,
}

// An Instruction without the input and output information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DInstruction {
    Op(OpId),
    GetAttr(ParameterId, ShardingForm),
    Placeholder(PlaceholderId, ShardingForm),
    Output,
    Communication(Collective)
}

impl HoareTriple {
    fn get_cost(&self, profiler: &Profiler, sharding_ratio: &[f64]) -> f64 {
        match self.instruction {
            DInstruction::Op(op_id) => {
                3. * profiler.get_computation_time(&self, &profiler.module_info[op_id], sharding_ratio)
            },
            DInstruction::GetAttr(_, _) => 1e-6,
            DInstruction::Placeholder(_, _) => 1e-6,
            DInstruction::Output => 1e-6,
            DInstruction::Communication(c) => {
                let original_size = if let Property::HasTensor(tensor_id, _) = self.pre_conditions[0] { // TODO: how to properly get the shape information?
                    profiler.rgraph[tensor_id].size() as f64
                } else {
                    unreachable!()
                };

                let maximum_size = original_size * sharding_ratio.iter().cloned().map(FloatOrd).max().unwrap().0;

                profiler.get_collective_time(maximum_size, c) + c.conjugate().map(|c| profiler.get_collective_time(maximum_size, c)).unwrap_or(0.0)
            },
        }
    }
}

impl Display for DInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DInstruction::Op(op) => write!(f, "Op({})", op.0),
            DInstruction::GetAttr(attr, form) => write!(f, "GetAttr({}, {:?})", attr.0, form),
            DInstruction::Placeholder(input, form) => write!(f, "Placeholder({}, {:?})", input.0, form),
            DInstruction::Output => write!(f, "Output"),
            DInstruction::Communication(collective) => write!(f, "Communication({:?})", collective),
        }
    }
}

#[derive(Debug, Clone)]
struct Profiler<'r, 'm, 'c> {
    rgraph: &'r RGraph,
    module_info: &'m ModuleInfo,
    cluster_info: &'c ClusterInfo,
}

impl<'r, 'm, 'c> Profiler<'r, 'm, 'c> {
    fn get_collective_time(&self, size: f64, collective: Collective) -> f64 {
        match collective {
            Collective::AllReduce => size / self.cluster_info.all_reduce_bandwidth,
            Collective::AllGather(_) => size / self.cluster_info.all_gather_bandwidth,
            Collective::AllToAll(_, _) => size / self.cluster_info.all_to_all_bandwidth,
            Collective::ReduceScatter(_) => size / self.cluster_info.reduce_scatter_bandwidth,
            Collective::DynamicSlice(_) => 0.,
            Collective::Replicate => 0.
        }
    }

    fn get_computation_time(&self, triple: &HoareTriple, op: &Op, sharding_ratio: &[f64]) -> f64 {
        let sharded = triple.pre_conditions.iter().any(|p| matches!(p, Property::HasTensor(_, TensorRelation::Gather(_))));
        let time_on_devices = (0..self.cluster_info.n_devices).map(|device_index| {
            let flops = (op.flops)() * sharded.then(|| sharding_ratio[device_index]).unwrap_or(1.);
            flops / self.cluster_info.device_flops[device_index]
        });

        time_on_devices.map(FloatOrd).max().unwrap().0
    }
}

#[derive(Default, Debug, Clone)]
struct Program {
    triples: Vec<HoareTriple>,
    properties: BTreeSet<Property>, // active properties whose corresponding tensors may still be used by future instructions
    cost: f64,
    ecost: f64,
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "length: {}, cost: {}, ecost: {}", self.triples.len(), self.cost, self.ecost)?;

        writeln!(f, "=== active properties ===")?;
        for property in &self.properties {
            writeln!(f, "{property}")?;
        }
        writeln!(f, "=== triples ===")?;
        for triple in &self.triples {
            writeln!(f, "{triple}")?;
        }
        Ok(())
    }
}

impl Program {
    fn empty(properties: impl IntoIterator<Item=Property>) -> Program {
        Program { properties: properties.into_iter().collect(), ..Default::default() }
    }

    fn with_a_new_triple(&self, triple: &HoareTriple, profiler: &Profiler) -> Program {
        let mut triples = self.triples.clone();
        triples.push(triple.clone());

        let mut properties = self.properties.iter()
            .filter(|p| !triple.negative_post_conditions.contains(p))
            .chain(triple.post_conditions.iter())
            .cloned()
            .collect();

        let cost = self.cost + triple.get_cost(profiler, &[0.25; 4]);
        let ecost = 0.0;

        Program { triples, properties, cost, ecost }
    }

    fn find_available_triples<'s, 't: 's>(&'s self, triples: &'t IndexedHoareTriples) -> Vec<HoareTripleId> {
        let candidates: BTreeSet<_> = self.properties.iter().flat_map(|p| triples.get_triples_with_pre_condition(*p)).copied().collect();

        candidates.into_iter().filter(|triple_id| {
            let triple = &triples[*triple_id];
            triple.pre_conditions.iter().all(|p| self.properties.contains(p)) && triple.post_conditions.iter().any(|p| !self.properties.contains(p))
        }).collect()
    }

    fn is_complete(&self) -> bool {
        self.properties.iter().any(|p| *p == Property::Finished)
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

new_usize_type!(pub, HoareTripleId);

#[derive(Debug)]
struct IndexedHoareTriples {
    triples: Vec<HoareTriple>,
    pre_condition_index: BTreeMap<Property, Vec<HoareTripleId>>,
    post_condition_index: BTreeMap<Property, Vec<HoareTripleId>>,
}

impl IndexedHoareTriples {
    fn new(triples: &[HoareTriple]) -> Self {
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

        IndexedHoareTriples { triples: triples.iter().cloned().collect(), pre_condition_index, post_condition_index }
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

impl Index<HoareTripleId> for IndexedHoareTriples {
    type Output = HoareTriple;

    fn index(&self, index: HoareTripleId) -> &Self::Output {
        &self.triples[index.0]
    }
}

fn a_star(triples: &[HoareTriple], initial_properties: &[Property], profiler: &Profiler) -> Program {
    let mut heap = BinaryHeap::new();
    let mut best_program: Option<Program> = None;
    let mut property_cache: BTreeMap<BTreeSet<Property>, f64> = BTreeMap::new();

    heap.push(ProgramHeapEntry::new(Program::empty(initial_properties.iter().cloned())));
    property_cache.insert(initial_properties.iter().cloned().collect(), 0.);

    let triples = IndexedHoareTriples::new(triples);

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

        eprintln!("{program}");
        if program.is_complete() {
            if best_program.as_ref().map(|p| p.cost > program.cost).unwrap_or(true) {
                best_program = Some(program);
            }
        } else {
            for triple_id in program.find_available_triples(&triples) {
                let new_program = program.with_a_new_triple(&triples[triple_id], profiler);
                if let Some(&cached_cost) = property_cache.get(&new_program.properties) && cached_cost <= new_program.cost {
                    continue
                }
                property_cache.insert(new_program.properties.clone(), new_program.cost);
                heap.push(ProgramHeapEntry::new(new_program));
            }
        }

        ticker.tick();
    }

    eprintln!("===== Result =====\n\n{}", best_program.as_ref().unwrap());

    best_program.unwrap()
}

#[derive(Debug, Default)]
pub struct RGraph {
    nodes: Vec<RNode>,
    tensors: Vec<RTensor>,
}

#[derive(Debug, Default)]
struct ModuleInfo {
    placeholders: Vec<String>,
    parameters: Vec<String>,
    ops: Vec<Op>
}

impl Index<OpId> for ModuleInfo {
    type Output = Op;

    fn index(&self, index: OpId) -> &Self::Output {
        &self.ops[index.0]
    }
}

impl Index<PlaceholderId> for ModuleInfo {
    type Output = String;

    fn index(&self, index: PlaceholderId) -> &Self::Output {
        &self.placeholders[index.0]
    }
}

impl Index<ParameterId> for ModuleInfo {
    type Output = String;

    fn index(&self, index: ParameterId) -> &Self::Output {
        &self.parameters[index.0]
    }
}

#[derive(Debug)]
pub struct RNode {
    inputs: SVec<RTensorId>,
    outputs: SVec<RTensorId, 1>,
    instruction: RInstruction,
}

// An instruction in the reference graph without the input and output information
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RInstruction {
    Op(OpId),
    GetAttr(ParameterId),
    Placeholder(PlaceholderId),
    Output
}

#[derive(Debug)]
pub struct RTensor {
    producer: RNodeId,
    consumers: SVec<RNodeId>,

    shape: SVec<usize>,
    communicatable: bool, // hints automatically generated for certain operatios (outputs of adaptive nodes are not communicatble), can be override by user annotation
}

impl RTensor {
    fn n_dims(&self) -> u8 {
        self.shape.len() as _
    }

    fn size(&self) -> usize {
        self.shape.iter().product()
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

struct OpCodegenContext<'py> {
    py: Python<'py>,
}

struct Op {
    py_name: String,
    codegen: Box<dyn Fn(&mut OpCodegenContext)>,
    flops: Box<dyn Fn() -> f64>, // todo
}

impl Debug for Op {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Op")
            .field("py_name", &self.py_name)
            .finish()
    }
}

struct ParserContext<'py, 'g, 'm, 'r> {
    py: Python<'py>,
    graph: &'g mut RGraph,
    module_info: &'m mut ModuleInfo,
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
}

fn initialize_parsing_handlers(py: Python) -> PyResult<BTreeMap<*mut (), &'static dyn Fn(ParserContext, PyObject) -> PyResult<()>>> {
    let mut parsing_handlers: BTreeMap<*mut (), &'static dyn Fn(ParserContext, PyObject) -> PyResult<()>> = BTreeMap::new();

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
        let op_id = OpId(ctx.module_info.ops.len());

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            shape: output_shape.clone().into(),
            communicatable: true
        });

        ctx.graph.nodes.push(RNode {
            inputs: smallvec![input_input_tensor_id, input_weight_tensor_id, input_bias_tensor_id],
            outputs: smallvec![tensor_id],
            instruction: RInstruction::Op(op_id)
        });

        ctx.module_info.ops.push(Op {
            py_name: "torch.nn.functional.linear".to_string(),
            codegen: Box::new(|ctx: &mut OpCodegenContext| {
                todo!()
            }),
            flops: Box::new({
                let reduction_length = ctx.graph[input_input_tensor_id].shape.last().copied().unwrap();
                move || {
                    3. * output_shape.iter().product::<usize>() as f64 * reduction_length as f64
                }
            })
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
        let op_id = OpId(ctx.module_info.ops.len());

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            shape: input_input_tensor.shape.clone(),
            communicatable: false
        });

        ctx.graph.nodes.push(RNode {
            inputs: smallvec![input_input_tensor_id],
            outputs: smallvec![tensor_id],
            instruction: RInstruction::Op(op_id)
        });

        ctx.module_info.ops.push(Op {
            py_name: "torch.sigmoid".to_string(),
            codegen: Box::new(|ctx: &mut OpCodegenContext| {
                todo!()
            }),
            flops: Box::new({
                let size = ctx.graph[input_input_tensor_id].shape.iter().product::<usize>();
                move || {
                    3. * size as f64
                }
            })
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
        let input_input_tensor = &ctx.graph[input_input_tensor_id];

        let node_id = RNodeId(ctx.graph.nodes.len());
        let tensor_id = RTensorId(ctx.graph.tensors.len());
        let op_id = OpId(ctx.module_info.ops.len());

        ctx.graph.tensors.push(RTensor {
            producer: node_id,
            consumers: smallvec![],
            shape: smallvec![],
            communicatable: false
        });

        ctx.graph.nodes.push(RNode {
            inputs: smallvec![input_input_tensor_id],
            outputs: smallvec![tensor_id],
            instruction: RInstruction::Op(op_id)
        });

        ctx.module_info.ops.push(Op {
            py_name: "torch.sum".to_string(),
            codegen: Box::new(|ctx: &mut OpCodegenContext| {
                todo!()
            }),
            flops: Box::new({
                let size = ctx.graph[input_input_tensor_id].shape.iter().product::<usize>();
                move || {
                    size as f64
                }
            })
        });

        ctx.graph[input_input_tensor_id].consumers.push(node_id);
        ctx.results[py_id] = Some(EvalResult::Tensor(tensor_id));
        Ok(())
    }

    Ok(parsing_handlers)
}

macro_rules! py_dict {
    ($py:expr, $($key:ident => $value:expr),*) => {{
        let dict = PyDict::new($py);
        $(
            dict.set_item($py, stringify!($key), &$value).unwrap();
        )*
        dict
    }}
}

fn load_fx_graph(py: Python, py_graph_module: PyObject, py_input_shape_dict: PyObject) -> PyResult<(RGraph, ModuleInfo)> {
    let mut graph = RGraph::default();
    let mut module_info = ModuleInfo::default();

    let parsing_handlers = initialize_parsing_handlers(py)?;

    let n_nodes = py_graph_module.getattr(py, "graph")?.getattr(py, "nodes")?.len(py)?;

    let mut results: Vec<Option<EvalResult>> = vec![None; n_nodes];

    for py_node in py_graph_module.getattr(py, "graph")?.getattr(py, "nodes")?.iter(py)? {
        let py_node = py_node?;
        let op_str: String = py_node.getattr(py, "op")?.extract(py)?;
        let py_id: usize = py_node.getattr(py, "meta")?.get_item(py, "id")?.extract(py)?;

        // memo when adding a node:
        // if the node has input, link the consumer of the inputs
        // if the node has output, set the result

        match &op_str[..] {
            "placeholder" => {
                let placeholder_id = PlaceholderId(module_info.placeholders.len());
                let name: String = py_node.getattr(py, "target")?.extract(py)?;
                let shape: Vec<usize> = py_input_shape_dict.get_item(py, &name)?.extract(py)?;

                module_info.placeholders.push(name);

                let node_id = RNodeId(graph.nodes.len());
                let tensor_id = RTensorId(graph.tensors.len());

                graph.nodes.push(RNode {
                    inputs: smallvec![],
                    outputs: smallvec![tensor_id],
                    instruction: RInstruction::Placeholder(placeholder_id),
                });

                graph.tensors.push(RTensor {
                    producer: node_id,
                    consumers: smallvec![],
                    shape: shape.into(),
                    communicatable: false
                });

                results[py_id] = Some(EvalResult::Tensor(tensor_id));
            },

            "get_attr" => {
                let parameter_id = ParameterId(module_info.parameters.len());
                let name: String = py_node.getattr(py, "target")?.extract(py)?;
                module_info.parameters.push(name);

                let shape: Vec<usize> = py.eval(
                    "get_shape_of_param_or_buffer(graph_module, node)",
                    None, Some(&py_dict!(py, graph_module => py_graph_module, node => py_node))
                )?.extract(py)?;

                let node_id = RNodeId(graph.nodes.len());
                let tensor_id = RTensorId(graph.tensors.len());

                graph.nodes.push(RNode {
                    inputs: smallvec![],
                    outputs: smallvec![tensor_id],
                    instruction: RInstruction::GetAttr(parameter_id),
                });

                graph.tensors.push(RTensor {
                    producer: node_id,
                    consumers: smallvec![],
                    shape: shape.into(),
                    communicatable: false
                });

                results[py_id] = Some(EvalResult::Tensor(tensor_id));
            },

            "call_function" => {
                let ctx = ParserContext {
                    py,
                    graph: &mut graph,
                    module_info: &mut module_info,
                    results: &mut results
                };

                parsing_handlers[&(py_node.getattr(py, "target")?.as_ptr() as _)](ctx, py_node)?;
            },

            "call_method" => {
                let ctx = ParserContext {
                    py,
                    graph: &mut graph,
                    module_info: &mut module_info,
                    results: &mut results
                };

                todo!()
            }

            "output" => {
                if graph.nodes.iter().any(|node| node.instruction == RInstruction::Output) {
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

    Ok((graph, module_info))
}

fn analyze_rgraph(rgraph: &RGraph, module_info: &ModuleInfo) -> Vec<HoareTriple> {
    let mut triples = vec![];

    let mut add_triple = |pre_conditions, post_conditions, instruction| {
        triples.push(HoareTriple {
            pre_conditions,
            post_conditions,
            negative_post_conditions: vec![],
            instruction,
        });
    };

    // basics: Placeholder, GetAttr, Output, and identity for ops
    for (node_id, node) in rgraph.nodes.iter().enumerate() {
        let node_id = RNodeId(node_id);

        match node.instruction {
            RInstruction::Placeholder(placeholder_id) => {
                let tensor_id = node.outputs[0];

                add_triple(
                    smallvec![],
                    smallvec![Property::identity(tensor_id)],
                    DInstruction::Placeholder(placeholder_id, ShardingForm::Unsharded)
                );

                for dim in 0..rgraph[tensor_id].n_dims() {
                    add_triple(
                        smallvec![],
                        smallvec![Property::gather(tensor_id, dim)],
                        DInstruction::Placeholder(placeholder_id, ShardingForm::Sharded(dim))
                    );
                }
            }

            RInstruction::GetAttr(parameter_id) => {
                let tensor_id = node.outputs[0];

                add_triple(
                    smallvec![],
                    smallvec![Property::identity(tensor_id)],
                    DInstruction::GetAttr(parameter_id, ShardingForm::Unsharded)
                );

                for dim in 0..rgraph[tensor_id].n_dims() {
                    add_triple(
                        smallvec![],
                        smallvec![Property::gather(tensor_id, dim)],
                        DInstruction::GetAttr(parameter_id, ShardingForm::Sharded(dim))
                    );
                }
            }

            RInstruction::Output => {
                let tensor_id = node.inputs[0];

                add_triple(
                    smallvec![Property::reduce(tensor_id)],
                    smallvec![Property::Finished],
                    DInstruction::Output
                );
            }

            RInstruction::Op(op_id) => {
                add_triple(
                    node.inputs.iter().map(|&tensor_id| Property::identity(tensor_id)).collect(),
                    node.outputs.iter().map(|&tensor_id| Property::identity(tensor_id)).collect(),
                    DInstruction::Op(op_id)
                );
            }
        }
    }

    // communication
    for (tensor_id, tensor) in rgraph.tensors.iter().enumerate() {
        let tensor_id = RTensorId(tensor_id);

        if !tensor.communicatable {
            continue;
        }

        for dim in 0..tensor.n_dims() {
            add_triple(
                smallvec![Property::gather(tensor_id, dim)],
                smallvec![Property::identity(tensor_id)],
                DInstruction::Communication(Collective::AllGather(dim))
            );

            add_triple(
                smallvec![Property::identity(tensor_id)],
                smallvec![Property::gather(tensor_id, dim)],
                DInstruction::Communication(Collective::DynamicSlice(dim))
            );
        }

        add_triple(
            smallvec![Property::reduce(tensor_id)],
            smallvec![Property::identity(tensor_id)],
            DInstruction::Communication(Collective::AllReduce)
        );

        for i in 0..tensor.n_dims() {
            for j in 0..tensor.n_dims() {
                if i != j {
                    add_triple(
                        smallvec![Property::gather(tensor_id, i)],
                        smallvec![Property::gather(tensor_id, j)],
                        DInstruction::Communication(Collective::AllToAll(j, i))
                    );
                }
            }
        }
    }

    macro_rules! for_each_op {
        ($op_name: expr, |$node_id: ident, $node: ident, $op_id: ident| $body: block) => {{
            for (node_id, $node) in rgraph.nodes.iter().enumerate() {
                let $node_id = RNodeId(node_id);
                if let RInstruction::Op($op_id) = $node.instruction && &module_info[$op_id].py_name == $op_name {
                    $body
                }
            }
        }}
    }

    // Linear
    for_each_op!("torch.nn.functional.linear", |node_id, node, op_id| {
        // data parallelism
        for dim in 0..rgraph[node.inputs[0]].n_dims() - 1 {
            add_triple(
                smallvec![
                    Property::gather(node.inputs[0], dim),
                    Property::identity(node.inputs[1]),
                    Property::identity(node.inputs[2]),
                ],
                smallvec![Property::gather(node.outputs[0], dim)],
                DInstruction::Op(op_id)
            );
        }

        // feature partition
        add_triple(
            smallvec![
                Property::identity(node.inputs[0]),
                Property::gather(node.inputs[1], 0),
                Property::gather(node.inputs[2], 0),
            ],
            smallvec![Property::gather(node.outputs[0], rgraph[node.outputs[0]].n_dims() - 1)],
            DInstruction::Op(op_id)
        );

        // reduction?
        // this requires arithemetic replacement (change to matmul + allreduce + add)
        // we also hit Rust aliasing rule here as the loop already borrows the graph
    });

    // Sigmoid
    for_each_op!("torch.sigmoid", |node_id, node, op_id| {
        for dim in 0..rgraph[node.inputs[0]].n_dims() {
            add_triple(
                smallvec![Property::gather(node.inputs[0], dim)],
                smallvec![Property::gather(node.outputs[0], dim)],
                DInstruction::Op(op_id)
            );
        }
    });

    // Sum
    for_each_op!("torch.sum", |node_id, node, op_id| {
        for dim in 0..rgraph[node.inputs[0]].n_dims() {
            add_triple(
                smallvec![Property::gather(node.inputs[0], dim)],
                smallvec![Property::reduce(node.outputs[0])],
                DInstruction::Op(op_id)
            );
        }

        add_triple(
            smallvec![Property::reduce(node.inputs[0])],
            smallvec![Property::reduce(node.outputs[0])],
            DInstruction::Op(op_id)
        );
    });

    triples
}

mod heuristics {
    use super::*;

    // each Op can only be computed once
    pub fn compute_only_once(triples: &mut [HoareTriple], _rgraph: &RGraph) -> Vec<Property> {
        let mut default_properties = vec![];
        for triple in triples {
            if let DInstruction::Op(op_id) = triple.instruction {
                triple.pre_conditions.push(Property::AllowComputation(op_id));
                triple.negative_post_conditions.push(Property::AllowComputation(op_id));
                default_properties.push(Property::AllowComputation(op_id));
            }
        }
        default_properties
    }

    // communication must happen in order
    pub fn ordered_communication(triples: &mut [HoareTriple], rgraph: &RGraph) -> Vec<Property> {
        let mut default_properties = vec![];
        for triple in triples {
            if let DInstruction::Communication(_) = triple.instruction {
                if let Property::HasTensor(tensor_id, _) = triple.post_conditions[0] {
                    triple.pre_conditions.push(Property::AllowCommunication(tensor_id));
                    for i in 0..=tensor_id.0 {
                        if rgraph[RTensorId(i)].communicatable {
                            triple.negative_post_conditions.push(Property::AllowCommunication(RTensorId(i)));
                        }
                    }
                    default_properties.push(Property::AllowCommunication(tensor_id));
                } else {
                    unreachable!();
                }
            }
        }
        default_properties
    }

    // placeholder must happen in order
    pub fn ordered_placeholder(triples: &mut [HoareTriple], rgraph: &RGraph) -> Vec<Property> {
        let mut default_properties = vec![];
        for triple in triples {
            if let DInstruction::Placeholder(placeholder_id, _) = triple.instruction {
                triple.pre_conditions.push(Property::AllowPlaceholder(placeholder_id));
                for i in 0..=placeholder_id.0 {
                    triple.negative_post_conditions.push(Property::AllowPlaceholder(PlaceholderId(i)));
                }
                default_properties.push(Property::AllowPlaceholder(placeholder_id));
            }
        }
        default_properties
    }

    // get attr must happen in order
    pub fn ordered_get_attr(triples: &mut [HoareTriple], rgraph: &RGraph) -> Vec<Property> {
        let mut default_properties = vec![];
        for triple in triples {
            if let DInstruction::GetAttr(parameter_id, _) = triple.instruction {
                triple.pre_conditions.push(Property::AllowGetAttr(parameter_id));
                for i in 0..=parameter_id.0 {
                    triple.negative_post_conditions.push(Property::AllowGetAttr(ParameterId(i)));
                }
                default_properties.push(Property::AllowGetAttr(parameter_id));
            }
        }
        default_properties
    }
}

#[derive(Debug)]
struct ClusterInfo {
    n_devices: usize,
    device_flops: Vec<f64>,
    all_reduce_bandwidth: f64,
    all_gather_bandwidth: f64,
    all_to_all_bandwidth: f64,
    reduce_scatter_bandwidth: f64,
}


