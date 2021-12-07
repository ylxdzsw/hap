use std::str::FromStr;

use super::SVec;

#[derive(Clone, Debug)]
pub struct Node {
    pub origin_id: usize, // the index of the original node
    pub op_kind: OpKind,
    pub inputs: SVec<TensorIndex>,
    pub outputs: SVec<TensorIndex>,
    pub signatures: Vec<Signature>
}

#[derive(Clone, Default, Debug)]
pub struct Tensor {
    pub size: usize,
    pub producer: NodeIndex,
    pub consumers: SVec<NodeIndex>,
    pub forms: Vec<Form> // all possible forms. Form::Undecided is not listed
}

#[derive(Clone, Default, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub tensors : Vec<Tensor>
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Form { Full, Gather(u8), Reduce, Replicate, Undecided }

impl Default for Form {
    fn default() -> Form { Form::Undecided }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum OpKind { Placeholder, GetAttr, CallFunction, CallMethod, Output }

impl ToString for OpKind {
    fn to_string(&self) -> String {
        match self {
            OpKind::Placeholder => "placeholder".to_string(),
            OpKind::GetAttr => "get_attr".to_string(),
            OpKind::CallFunction => "call_function".to_string(),
            OpKind::CallMethod => "call_method".to_string(),
            OpKind::Output => "output".to_string(),
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
    pub output_forms: SVec<Form>,

    pub cost: f64,
}

pub type NodeIndex = usize;
pub type TensorIndex = usize;
pub type SignatureIndex = usize;
