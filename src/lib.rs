#![allow(unused)]

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
        load_fx_graph(py, py_graph_module);
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

        impl From<usize> for $type_name {
            fn from(x: usize) -> $type_name {
                $type_name(x)
            }
        }
    }
}

pub(crate) use new_usize_type;

new_usize_type!(pub, RNodeId);
new_usize_type!(pub, RTensorId);

new_usize_type!(pub, PyOpCodeId);
new_usize_type!(pub, PyParameterId);
new_usize_type!(pub, PyInputId);

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Instruction {
    Op(PyOpCodeId),
    GetAttr(PyParameterId),
    Placeholder(PyInputId),
    Output,
    Communication(RTensorId, Collective)
}

impl Instruction {
    fn get_cost(&self, profiler: &Profiler) -> f64 {
        match self {
            Instruction::Op(_) => 1.0,
            Instruction::GetAttr(_) => 0.1,
            Instruction::Placeholder(_) => 0.1,
            Instruction::Output => 0.1,
            Instruction::Communication(_, _) => 1.0,
        }
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Op(op) => write!(f, "Op({})", op.0),
            Instruction::GetAttr(attr) => write!(f, "GetAttr({})", attr.0),
            Instruction::Placeholder(input) => write!(f, "Placeholder({})", input.0),
            Instruction::Output => write!(f, "Output"),
            Instruction::Communication(tensor_id, collective) => write!(f, "Communication({}, {:?})", tensor_id.0, collective),
        }
    }
}

struct Profiler;

#[derive(Default, Debug, Clone)]
struct Program {
    triples: Vec<HoareTriple>,
    properties: BTreeSet<Property>,
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

        writeln!(f, "=== properties ===")?;
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

        let next_communicatable_id = if let Instruction::Communication(tensor_id, _) = triple.instruction {
            tensor_id + 1
        } else {
            self.next_communicatable_id
        };

        let next_free_id = match triple.instruction {
            Instruction::GetAttr(_) | Instruction::Placeholder(_) => triple.post_conditions[0].tensor_id + 1,
            _ => self.next_free_id,
        };

        Program { triples, properties, cost, ecost, next_communicatable_id, next_free_id }
    }

    fn find_available_triples<'s, 't: 's>(&'s self, triples: &'t [HoareTriple]) -> Vec<&'t HoareTriple> {
        triples.iter().filter(|triple| {
            match triple.instruction {
                Instruction::GetAttr(_) | Instruction::Placeholder(_) if triple.post_conditions[0].tensor_id != self.next_free_id => return false,
                Instruction::Communication(tensor_id, _) if tensor_id < self.next_communicatable_id => return false,
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

fn load_fx_graph(py: Python, py_graph_module: PyObject) -> PyResult<Vec<HoareTriple>> {
    let mut triples = Vec::new();

    for py_node in py_nodes.iter(py) {
        let op =

        let name = py_node.getattr(py, "attr_name")?.extract::<String>(py)?;
    }


    eprint!("{}", py_nodes.len(py));

    Ok(triples)
}
