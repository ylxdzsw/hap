#![allow(unused)]

use std::ops::Index;
use std::sync::{atomic::AtomicBool, Arc};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt::Display;
use std::cmp::Ordering;
use float_ord::FloatOrd;
use oh_my_rust::*;
use cpython::{PyResult, PyTuple, ToPyObject, PythonObject, ObjectProtocol, Python, PyList, PyObject, PyDict};
use smallvec::{SmallVec, smallvec};

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
    m.add(py, "main", cpython::py_fn!(py, main(py_graph_module: PyObject, py_config: PyObject) -> PyResult<PyList> {
        let triples = load_fx_graph(py, py_graph_module)?;
        for triple in &triples {
            eprintln!("{}", triple);
        }
        a_star(&triples, &Profiler);
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

pub(crate) use new_usize_type;

// Some names:
// R stands for Reference (the nodes and tensors in the orignal single card graph)
// D stands for Distributed (the nodes and tensors in the SIMD graph)
// Op is a curried pytorch operator with non-tensor parameters filled in
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

#[derive(Debug, Clone)]
struct HoareTriple {
    pre_conditions: SVec<Property>,
    post_conditions: SVec<Property, 1>,
    instruction: Instruction,
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
        write!(f, "}}")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Property {
    tensor_id: RTensorId,
    relation: PropertyRelation,
}

impl Display for Property {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}|{:?}", self.tensor_id.0, self.relation)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PropertyRelation {
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
enum Instruction {
    Op(OpId), // we use the name "op" to refer to a pytorch operator with non-tensor parameters filled in
    GetAttr(ParameterId, ShardingForm),
    Placeholder(PlaceholderId, ShardingForm),
    Output,
    Communication(Collective)
}

impl Instruction {
    fn get_cost(&self, profiler: &Profiler) -> f64 {
        match self {
            Instruction::Op(_) => 1.0,
            Instruction::GetAttr(_, _) => 0.1,
            Instruction::Placeholder(_, _) => 0.1,
            Instruction::Output => 0.1,
            Instruction::Communication(_) => 1.0,
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Op(op) => write!(f, "Op({})", op.0),
            Instruction::GetAttr(attr, form) => write!(f, "GetAttr({}, {:?})", attr.0, form),
            Instruction::Placeholder(input, form) => write!(f, "Placeholder({}, {:?})", input.0, form),
            Instruction::Output => write!(f, "Output"),
            Instruction::Communication(collective) => write!(f, "Communication({:?})", collective),
        }
    }
}

struct Profiler;

#[derive(Default, Debug, Clone)]
struct Program {
    triples: Vec<HoareTriple>,
    properties: BTreeSet<Property>, // active properties whose corresponding tensors may still be used by future instructions
    cost: f64,
    ecost: f64,

    next_communicatable_id: RTensorId,
    next_free_id: RTensorId,
}




impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "length: {}, cost: {}, ecost: {}", self.triples.len(), self.cost, self.ecost)?;
        writeln!(f, "next communicatable id: {}", self.next_communicatable_id.0);
        writeln!(f, "next free id: {}", self.next_free_id.0);

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
    fn with_a_new_triple(&self, triple: &HoareTriple, profiler: &Profiler) -> Program {
        let mut triples = self.triples.clone();
        triples.push(triple.clone());

        let mut properties = self.properties.clone();
        properties.extend(triple.post_conditions.iter().cloned());

        let cost = self.cost + triple.instruction.get_cost(profiler);
        let ecost = 0.0;

        let next_communicatable_id = if let Instruction::Communication(_) = triple.instruction {
            triple.post_conditions[0].tensor_id + 1
        } else {
            self.next_communicatable_id
        };

        let next_free_id = match triple.instruction {
            Instruction::GetAttr(_, _) | Instruction::Placeholder(_, _) => triple.post_conditions[0].tensor_id + 1,
            _ => self.next_free_id,
        };

        Program { triples, properties, cost, ecost, next_communicatable_id, next_free_id }
    }

    fn find_available_triples<'s, 't: 's>(&'s self, triples: &'t [HoareTriple]) -> Vec<&'t HoareTriple> {
        triples.iter().filter(|triple| {
            match triple.instruction {
                Instruction::GetAttr(_, _) | Instruction::Placeholder(_, _) if triple.post_conditions[0].tensor_id != self.next_free_id => return false,
                Instruction::Communication(_) if triple.post_conditions[0].tensor_id < self.next_communicatable_id => return false,
                _ => {}
            }

            triple.pre_conditions.iter().all(|p| self.properties.contains(p)) && (triple.post_conditions.iter().any(|p| !self.properties.contains(p)) || triple.instruction == Instruction::Output)
        }).collect()
    }

    fn is_complete(&self) -> bool {
        self.triples.iter().any(|p| p.instruction == Instruction::Output)
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

fn a_star(triples: &[HoareTriple], profiler: &Profiler) -> Program {
    let mut heap = BinaryHeap::new();

    let mut best_program: Option<Program> = None;

    heap.push(ProgramHeapEntry::new(Program::default()));

    while let Some(ProgramHeapEntry { program, .. }) = heap.pop() {
        if best_program.as_ref().map(|p| p.cost < program.cost).unwrap_or(false) {
            continue;
        }

        eprintln!("{program}");
        if program.is_complete() {
            if best_program.as_ref().map(|p| p.cost > program.cost).unwrap_or(true) {
                best_program = Some(program);
            }
        } else {
            for triple in program.find_available_triples(triples) {
                let new_program = program.with_a_new_triple(triple, profiler);
                heap.push(ProgramHeapEntry::new(new_program));
            }
        }
    }

    eprintln!("===== Result =====\n\n{}", best_program.as_ref().unwrap());

    best_program.unwrap()
}

#[derive(Debug)]
struct RGraph {
    nodes: Vec<RNode>,
    tensors: Vec<RTensor>,
}


#[derive(Debug)]
struct RNode {

}

#[derive(Debug)]
struct RTensor {

}

impl Index<RNodeId> for RGraph {
    type Output = RNode;

    fn index(&self, index: RNodeId) -> &Self::Output {
        &self.nodes[index.0]
    }
}

impl Index<RTensorId> for RGraph {
    type Output = RTensor;

    fn index(&self, index: RTensorId) -> &Self::Output {
        &self.tensors[index.0]
    }
}

struct OpCodegenContext<'py> {
    py: Python<'py>,
}

struct OpProfileContext {
}

struct Op {
    codegen: Box<dyn Fn(&mut OpCodegenContext)>,
    profile: Box<dyn Fn(&mut OpProfileContext)>
}

fn load_fx_graph(py: Python, py_graph_module: PyObject) -> PyResult<Vec<HoareTriple>> {
    macro_rules! py_meta {
        ($py_node: expr, $meta_name: expr) => { py_meta!($py_node, $meta_name, _) };
        ($py_node: expr, $meta_name: expr, $out_type: ty) => {{
            let val = ($py_node).getattr(py, "meta")?.call_method(py, "get", ($meta_name, ), None)?;
            if val.is_none(py) {
                None
            } else {
                Some(val.extract::<$out_type>(py)?)
            }
        }}
    }

    fn parse_relation(form: &str) -> PropertyRelation {
        match form {
            "gather_0" => PropertyRelation::Gather(0),
            "gather_1" => PropertyRelation::Gather(1),
            "gather_2" => PropertyRelation::Gather(2),
            "gather_3" => PropertyRelation::Gather(3),
            "full" => PropertyRelation::Identity,
            "reduce" => PropertyRelation::Reduce,
            _ => unreachable!()
        }
    }

    let mut triples = Vec::new();

    let mut next_placeholder_id = PlaceholderId(0);
    let mut next_parameter_id = ParameterId(0);
    let mut has_output = false;

    for py_node in py_graph_module.getattr(py, "graph")?.getattr(py, "nodes")?.iter(py)? {
        let py_node = py_node?;
        let op = py_node.getattr(py, "op")?.extract::<String>(py)?;
        let meta = py_node.getattr(py, "meta")?;
        let id = meta.get_item(py, "id")?.extract::<usize>(py)?;

        println!("{:?}, {:?}", op, meta);

        match &op[..] {
            "placeholder" => {
                let n_dims = meta.get_item(py, "output_shape")?.cast_as::<PyTuple>(py)?.len(py) as u8;
                for i in 0..n_dims {
                    triples.push(HoareTriple {
                        instruction: Instruction::Placeholder(next_placeholder_id, ShardingForm::Sharded(i)),
                        pre_conditions: smallvec![],
                        post_conditions: smallvec![Property {
                            tensor_id: RTensorId(meta.get_item(py, "id")?.extract::<usize>(py)?),
                            relation: PropertyRelation::Gather(i)
                        }],
                    })
                }
                triples.push(HoareTriple {
                    instruction: Instruction::Placeholder(next_placeholder_id, ShardingForm::Unsharded),
                    pre_conditions: smallvec![],
                    post_conditions: smallvec![Property {
                        tensor_id: RTensorId(meta.get_item(py, "id")?.extract::<usize>(py)?),
                        relation: PropertyRelation::Identity
                    }]
                });
                next_placeholder_id += 1;
            },

            "get_attr" => {
                let n_dims = meta.get_item(py, "output_shape")?.cast_as::<PyTuple>(py)?.len(py) as u8;
                for i in 0..n_dims {
                    triples.push(HoareTriple {
                        instruction: Instruction::GetAttr(next_parameter_id, ShardingForm::Sharded(i)),
                        pre_conditions: smallvec![],
                        post_conditions: smallvec![Property {
                            tensor_id: RTensorId(meta.get_item(py, "id")?.extract::<usize>(py)?),
                            relation: PropertyRelation::Gather(i)
                        }],
                    })
                }
                triples.push(HoareTriple {
                    instruction: Instruction::GetAttr(next_parameter_id, ShardingForm::Unsharded),
                    pre_conditions: smallvec![],
                    post_conditions: smallvec![Property {
                        tensor_id: RTensorId(meta.get_item(py, "id")?.extract::<usize>(py)?),
                        relation: PropertyRelation::Identity
                    }]
                });
                next_parameter_id += 1;
            },

            "call_function" | "call_method" => {
                let arg_dict: PyDict = py_meta!(py_node, "arg_dict").expect("no arg_dict in meta");
                let py_signatures: PyList = py_meta!(py_node, "signatures").expect("no signature found in meta");
                for py_signature in py_signatures.iter(py) {
                    let post_conditions =
                        smallvec![Property {
                            tensor_id: RTensorId(py_meta!(py_node, "id", usize).unwrap()),
                            relation: parse_relation(&py_signature.get_item(py, 1usize)?.extract::<Cow<str>>(py)?)
                        }];

                    let mut pre_conditions = smallvec![];
                    let arg_form_dict: PyDict = py_signature.get_item(py, 0usize)?.cast_into(py)?;
                    for (arg_name, arg_form) in arg_form_dict.items(py) {
                        let py_arg_node = arg_dict.as_object().get_item(py, arg_name)?;
                        if py_arg_node.hasattr(py, "meta")? { // is a node, not literal
                            let arg_tensor_id = py_meta!(py_arg_node, "id", usize).unwrap();
                            pre_conditions.push(Property {
                                tensor_id: RTensorId(arg_tensor_id),
                                relation: parse_relation(&arg_form.extract::<Cow<str>>(py)?)
                            });
                        }
                    }

                    triples.push(HoareTriple {
                        instruction: Instruction::Op(OpId(id)),
                        pre_conditions,
                        post_conditions,
                    });
                }

                let mut full_condition = smallvec![];
                for (arg_name, py_arg_node) in arg_dict.items(py) {
                    if py_arg_node.hasattr(py, "meta")? { // is a node, not literal
                        let arg_tensor_id = py_meta!(py_arg_node, "id", usize).unwrap();
                        full_condition.push(Property {
                            tensor_id: RTensorId(arg_tensor_id),
                            relation: PropertyRelation::Identity
                        });
                    }
                }
                triples.push(HoareTriple {
                    instruction: Instruction::Op(OpId(id)),
                    pre_conditions: full_condition,
                    post_conditions: smallvec![Property {
                        tensor_id: RTensorId(id),
                        relation: PropertyRelation::Identity
                    }]
                });
            },

            "output" => {
                if has_output {
                    panic!("Multiple outputs in the graph");
                }

                triples.push(HoareTriple {
                    instruction: Instruction::Output,
                    pre_conditions: smallvec![Property {
                        tensor_id: py_node.getattr(py, "args")?.get_item(py, 0)?.getattr(py, "meta")?.get_item(py, "id")?.extract::<usize>(py)?.into(),
                        relation: PropertyRelation::Reduce
                    }],
                    post_conditions: smallvec![],
                });

                has_output = true;
            },

            _ => unreachable!()
        }

        triples.push(HoareTriple {
            pre_conditions: smallvec![Property {
                tensor_id: RTensorId(id),
                relation: PropertyRelation::Reduce
            }],
            post_conditions: smallvec![Property {
                tensor_id: RTensorId(id),
                relation: PropertyRelation::Identity
            }],
            instruction: Instruction::Communication(Collective::AllReduce)
        });

        let n_dims = meta.get_item(py, "output_shape")?.cast_as::<PyTuple>(py)?.len(py) as u8;

        for i in 0..n_dims {
            triples.push(HoareTriple {
                pre_conditions: smallvec![Property {
                    tensor_id: RTensorId(id),
                    relation: PropertyRelation::Gather(i)
                }],
                post_conditions: smallvec![Property {
                    tensor_id: RTensorId(id),
                    relation: PropertyRelation::Identity
                }],
                instruction: Instruction::Communication(Collective::AllGather(i))
            });

            triples.push(HoareTriple {
                pre_conditions: smallvec![Property {
                    tensor_id: RTensorId(id),
                    relation: PropertyRelation::Identity
                }],
                post_conditions: smallvec![Property {
                    tensor_id: RTensorId(id),
                    relation: PropertyRelation::Gather(i)
                }],
                instruction: Instruction::Communication(Collective::DynamicSlice(i))
            });
        }

        for i in 0..n_dims {
            for j in 0..n_dims {
                if i != j {
                    triples.push(HoareTriple {
                        pre_conditions: smallvec![Property {
                            tensor_id: RTensorId(id),
                            relation: PropertyRelation::Gather(i)
                        }],
                        post_conditions: smallvec![Property {
                            tensor_id: RTensorId(id),
                            relation: PropertyRelation::Gather(j)
                        }],
                        instruction: Instruction::Communication(Collective::AllToAll(j, i))
                    });
                }
            }
        }

    }


    Ok(triples)
}

struct AnnotationContext {

}
