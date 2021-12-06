#![feature(const_generics_defaults)]
#![feature(label_break_value)]

#![allow(unused)]

use cpython::{PyResult, PyTuple, ToPyObject, PythonObject, ObjectProtocol};
use smallvec::SmallVec;

type SVec<T, const N: usize = 4> = SmallVec<[T; N]>;

cpython::py_module_initializer!(spmd, |py, m| {
    #[allow(clippy::manual_strip)]
    m.add(py, "spmd", cpython::py_fn!(py, example() -> PyResult<PyTuple> {
        Ok((2, ).to_py_object(py))
    }))?;

    Ok(())
});

