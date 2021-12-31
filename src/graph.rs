use std::{str::FromStr, fmt::Display};

use oh_my_rust::*;

use super::{SVec, smallvec};

#[derive(Clone, Debug)]
pub struct Node {
    pub origin_id: usize, // the index of the original node
    pub op_kind: OpKind,
    pub inputs: SVec<TensorIndex>,
    pub outputs: SVec<TensorIndex>,
    pub signatures: Vec<Signature>,

    pub input_names: SVec<String>, // original node name of the inputs
    pub companions: SVec<usize>, // the origin_id of adaptive nodes in the output order

    pub flops: u64,
    pub name: String,
}

#[derive(Clone, Default, Debug)]
pub struct Tensor {
    pub size: u64,
    pub producer: NodeIndex,
    pub consumers: SVec<NodeIndex>,
    pub producer_forms: Vec<Form>, // all possible forms that could be produced
    pub consumer_forms: Vec<Form>, // all possible forms that could be consumed
}

#[derive(Clone, Default, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub tensors : Vec<Tensor>
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Form { Full, Gather(u8), Reduce, Replicate }

impl Display for Form {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Form::Full => write!(f, "full"),
            Form::Gather(i) => write!(f, "gather_{}", i),
            Form::Reduce => write!(f, "reduce"),
            Form::Replicate => write!(f, "replicate"),
        }
    }
}

impl FromStr for Form {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "full" => Ok(Form::Full),
            "reduce" => Ok(Form::Reduce),
            "replicate" => Ok(Form::Replicate),
            _ if s.starts_with("gather_") => {
                s[7..].parse().map(Form::Gather).map_err(|_| ())
            }
            _ => Err(())
        }
    }
}

impl Form {
    pub fn collective_reform(self, new_form: Form) -> Option<SVec<Collective, 2>> {
        match (self, new_form) {
            (a, b) if a == b => Some(smallvec![]),
            (Form::Full, Form::Gather(dim)) => Some(smallvec![Collective::DynamicSlice(dim)]),
            (Form::Gather(dim), Form::Full) => Some(smallvec![Collective::AllGather(dim)]),
            (Form::Gather(cat_dim), Form::Gather(split_dim)) => Some(smallvec![Collective::AllToAll(split_dim, cat_dim)]), // this must not be the same
            (Form::Reduce, Form::Full) => Some(smallvec![Collective::AllReduce]),
            (Form::Reduce, Form::Gather(dim)) => Some(smallvec![Collective::ReduceScatter(dim)]),
            (Form::Replicate, Form::Full) => Some(smallvec![Collective::Replicate]),
            (Form::Replicate, Form::Gather(dim)) => Some(smallvec![Collective::Replicate, Collective::DynamicSlice(dim)]),
            _ => None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpKind { Placeholder, GetAttr, CallFunction, CallMethod, Output }

impl Display for OpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpKind::Placeholder => write!(f, "placeholder"),
            OpKind::GetAttr => write!(f, "get_attr"),
            OpKind::CallFunction => write!(f, "call_function"),
            OpKind::CallMethod => write!(f, "call_method"),
            OpKind::Output => write!(f, "output"),
        }
    }
}

impl FromStr for OpKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "placeholder" => Ok(OpKind::Placeholder),
            "get_attr" => Ok(OpKind::GetAttr),
            "call_function" => Ok(OpKind::CallFunction),
            "call_method" => Ok(OpKind::CallMethod),
            "output" => Ok(OpKind::Output),
            _ => Err(())
        }
    }
}

#[derive(Clone, Default, Debug)]
pub struct Signature {
    pub input_forms: SVec<Form>,
    pub output_forms: SVec<Form>
}

crate::new_index_type!(pub, NodeIndex);
crate::new_index_type!(pub, TensorIndex);
crate::new_index_type!(pub, SignatureIndex);

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
    pub fn conjugate(self) -> Option<Self> {
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

impl Display for Collective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Collective::AllGather(dim) => write!(f, "all_gather_{}", dim),
            Collective::AllReduce => write!(f, "all_reduce"),
            Collective::ReduceScatter(dim) => write!(f, "reduce_scatter_{}", dim),
            Collective::AllToAll(split_dim, cat_dim) => write!(f, "all_to_all_{}_{}", split_dim, cat_dim),
            Collective::Replicate => write!(f, "replicate"),
            Collective::DynamicSlice(dim) => write!(f, "dynamic_slice_{}", dim),
        }
    }
}
