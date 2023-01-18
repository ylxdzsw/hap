
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

new_usize_type!(pub, CompKind);
new_usize_type!(pub, DTensorIndex);

#[derive(Clone)]
struct CompOp {
    kind: CompKind,
    inputs: Vec<DTensorIndex>
}

#[derive(Clone)]
struct CommOp {

}

#[derive(Clone)]
enum Op {
    Comp,
    Comm
}

#[derive(Clone)]
struct Program {
    ops: Vec<Op>
}

impl Program {

}
