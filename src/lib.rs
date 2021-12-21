#![feature(const_generics_defaults)]
#![feature(label_break_value)]

#![allow(unused)]

use std::{collections::BTreeMap, borrow::Cow, sync::atomic::AtomicBool};

use oh_my_rust::*;
use cpython::{PyResult, PyTuple, ToPyObject, PythonObject, ObjectProtocol, Python, PyList, PyObject, PyDict};
use smallvec::{SmallVec, smallvec};

use crate::graph::{Tensor, Node, TensorIndex, Signature, Form, Graph, NodeIndex, OpKind};

mod graph;
mod dp;

pub type SVec<T, const N: usize = 3> = SmallVec<[T; N]>;

static CTRLC_TRAPPED: AtomicBool = AtomicBool::new(false);
static CTRLC_RECEIVED: AtomicBool = AtomicBool::new(false);

cpython::py_module_initializer!(spmd, |py, m| {
    if !CTRLC_TRAPPED.load(std::sync::atomic::Ordering::Relaxed) {
        ctrlc::set_handler(|| {
            CTRLC_RECEIVED.store(true, std::sync::atomic::Ordering::Relaxed)
        }).unwrap();
    }

    #[allow(clippy::manual_strip)]
    m.add(py, "example", cpython::py_fn!(py, example() -> PyResult<PyTuple> {
        Ok((2, ).to_py_object(py))
    }))?;

    #[allow(clippy::manual_strip)]
    m.add(py, "spmd", cpython::py_fn!(py, spmd(py_nodes: PyList, profiler: PyObject, hints: PyDict) -> PyResult<PyTuple> {
        let graph = build_graph(py, py_nodes, &profiler, hints)?;
        dp::dp2(&graph);
        Ok((2, ).to_py_object(py))
    }))?;

    Ok(())
});

fn build_graph(py: Python, py_nodes: PyList, profiler: &PyObject, hints: PyDict) -> PyResult<Graph> {
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

    let n = py_nodes.len(py);

    let mut tensors: Vec<Tensor> = vec![];
    let mut nodes: Vec<Node> = Vec::with_capacity(n); // the node collection starts as one-to-one mapping of python nodes, then remove is_adaptive nodes before assigning node id (is_adaptive nodes are identified by empty signatures)

    for (id, py_node) in py_nodes.iter(py).enumerate() {
        debug_assert_eq!(id, py_meta!(py_node, "id").unwrap());

        let mut node = Node {
            origin_id: id,
            op_kind: py_node.getattr(py, "op")?.extract::<Cow<str>>(py)?.parse().unwrap(),
            inputs: Default::default(),
            outputs: Default::default(),
            signatures: Default::default(),
        };

        // adaptive nodes take a special path. It only sets its outputs as the same of the input and skip all others.
        if let Some(true) = py_meta!(py_node, "is_adaptive", bool) {
            let py_inputs = py_node.getattr(py, "all_input_nodes")?.cast_into::<PyList>(py)?;
            debug_assert_eq!(py_inputs.len(py), 1);
            let input_node_id = py_meta!(py_inputs.get_item(py, 0), "id", usize).unwrap();
            let input_node = &nodes[input_node_id];

            let input_index = if input_node.outputs.len() > 1 { // input is tuple, assume this node is getitem
                py_node.getattr(py, "args")?.get_item(py, 1usize)?.extract(py)?
            } else {
                0
            };
            node.outputs = smallvec![input_node.outputs[input_index]];
            nodes.push(node);
            continue
        }

        // generate output tensors and link them
        'gen_output_tensor: {
            if let OpKind::Output = node.op_kind {
                break 'gen_output_tensor
            }

            let meta_output_shape = py_meta!(py_node, "output_shape", PyTuple).unwrap();
            let output_shapes = if let Some(true) = py_meta!(py_node, "output_is_tuple") {
                meta_output_shape.iter(py).map(|x| x.extract(py)).collect::<PyResult<SVec<_>>>()?
            } else {
                smallvec![meta_output_shape]
            };
            for py_shape in output_shapes.iter() {
                let mut size = 4; // 4 bytes per element
                for s in py_shape.iter(py) {
                    size *= s.extract::<usize>(py)?
                }
                let tensor_index = TensorIndex(tensors.len());
                let mut tensor = Tensor { size, ..Default::default() };
                tensors.push(tensor);
                node.outputs.push(tensor_index);
            }
        }

        // input links
        for py_input_node in py_node.getattr(py, "all_input_nodes")?.cast_into::<PyList>(py)?.iter(py) {
            let input_node_id = py_meta!(py_input_node, "id", usize).unwrap();
            let input_node = &nodes[input_node_id];

            debug_assert_eq!(input_node.outputs.len(), 1);
            let input_tensor_index = input_node.outputs[0];

            debug_assert!(!node.inputs.contains(&input_tensor_index));
            node.inputs.push(input_tensor_index)
        }

        // set signatures
        match node.op_kind {
            OpKind::Output => { // the output node only has a Reduce signature
                let reduce_signature = Signature {
                    input_forms: smallvec![Form::Reduce],
                    ..Default::default()
                };
                node.signatures.push(reduce_signature);
            }
            OpKind::Placeholder => { // the placeholder only has a Full signature. We assume that each worker load the full batch then performing dynamic slicing
                let full_signature = Signature {
                    output_forms: smallvec![Form::Full],
                    ..Default::default()
                };
                node.signatures.push(full_signature);
            }
            OpKind::GetAttr => {
                let replication_signature = Signature { // this is the parameters analog of full signature
                    output_forms: smallvec![Form::Replicate],
                    ..Default::default()
                };
                node.signatures.push(replication_signature);
                for dim in 0..py_meta!(py_node, "output_shape", PyTuple).expect("no output shape in get_attr node").len(py) {
                    let gather_signature = Signature {
                        output_forms: smallvec![Form::Gather(dim as _)],
                        ..Default::default()
                    };
                    node.signatures.push(gather_signature)
                }
            }
            OpKind::CallFunction | OpKind::CallMethod => {
                // let flops = py_meta!(py_node, "flops").unwrap_or_default();

                let full_signature = Signature {
                    input_forms: node.inputs.iter().map(|_| Form::Full).collect(),
                    output_forms: node.outputs.iter().map(|_| Form::Full).collect(),
                    cost: 0., //comp_profiler.get_time(flops, None),
                };
                node.signatures.push(full_signature);

                let arg_dict: PyDict = py_meta!(py_node, "arg_dict").expect("no arg_dict in meta");
                let py_signatures: PyList = py_meta!(py_node, "signatures").expect("no signature found in meta");
                for py_signature in py_signatures.iter(py) {
                    let output_forms = if let Some(true) = py_meta!(py_node, "output_is_tuple") {
                        let out_forms: PyTuple = py_signature.get_item(py, 1usize)?.cast_into(py)?;
                        out_forms.iter(py).map(|out_form| out_form.extract(py).map(|x: Cow<str>| x.parse::<Form>().unwrap())).collect::<PyResult<_>>()?
                    } else {
                        smallvec![py_signature.get_item(py, 1usize)?.extract::<Cow<str>>(py)?.parse().unwrap()]
                    };

                    let mut input_forms: SVec<Option<Form>> = smallvec![None; node.inputs.len()];
                    let arg_form_dict: PyDict = py_signature.get_item(py, 0usize)?.cast_into(py)?;
                    for (arg_name, arg_form) in arg_form_dict.items(py) {
                        let py_arg_node = arg_dict.as_object().get_item(py, arg_name)?;
                        if py_arg_node.hasattr(py, "meta")? { // is a node, not literal
                            let arg_tensor_id = py_meta!(py_arg_node, "id", usize).unwrap();
                            let arg_tensor = &nodes[arg_tensor_id].outputs[0];
                            let i = node.inputs.iter().position(|x| x == arg_tensor).expect("input node not in inputs");
                            input_forms[i] = Some(arg_form.extract::<Cow<str>>(py)?.parse().unwrap());
                        }
                    }

                    let cost = 0.0; // comp_profiler.get_time(flops, &input_forms);
                    node.signatures.push(Signature {
                        input_forms: input_forms.into_iter().collect::<Option<SVec<_>>>().unwrap(),
                        output_forms, cost,
                    })
                }
            }
        }

        'process_hints: {
            let pyname = py_node.getattr(py, "name")?;
            let name: Cow<str> = pyname.extract(py)?;
            if let Some(forms) = hints.get_item(py, &name) {
                let forms: Vec<Form> = forms.extract::<PyList>(py)?.iter(py).map(|x| x.extract(py).map(|x: Cow<str>| x.parse::<Form>().unwrap())).collect::<PyResult<_>>()?;
                if node.outputs.len() != 1 {
                    warn!("hints for {} ignored because the node has multiple outputs", name);
                    break 'process_hints
                }
                node.signatures.retain(|signature| forms.contains(&signature.output_forms[0]));
                if node.signatures.is_empty() {
                    panic!("hints for {} cannot be satisfied!", name)
                }
            }
        }

        nodes.push(node);
    }

    let nodes: Vec<_> = nodes.into_iter().filter(|x| !x.signatures.is_empty()).collect();

    for (node_index, node) in nodes.iter().enumerate() {
        let node_index = NodeIndex(node_index);

        for &output in node.outputs.iter() {
            tensors[output.0].producer = node_index;
        }

        for &input in node.inputs.iter() {
            tensors[input.0].consumers.push(node_index);
        }

        for signature in node.signatures.iter() {
            for (&input_index, &input_form) in node.inputs.iter().zip(signature.input_forms.iter()) {
                let mut input_tensor = &mut tensors[input_index.0];
                if !input_tensor.consumer_forms.contains(&input_form) {
                    input_tensor.consumer_forms.push(input_form)
                }
            }

            for (&output_index, &output_form) in node.outputs.iter().zip(signature.output_forms.iter()) {
                let mut output_tensor = &mut tensors[output_index.0];
                if !output_tensor.producer_forms.contains(&output_form) {
                    output_tensor.producer_forms.push(output_form)
                }
            }
        }
    }

    Ok(Graph { nodes, tensors })
}

macro_rules! new_index_type {
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

pub(crate) use new_index_type;
