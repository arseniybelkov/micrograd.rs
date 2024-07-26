use crate::{Differentiable, Value};

pub trait Operation<'a, T: Differentiable> {
    fn forward(&mut self, lhs: &'a Value<'a, T>, rhs: &'a Value<'a, T>) -> Value<'a, T>;
    fn backward(&self, grad: T); 
}

pub(crate) enum Operand<'a, T: Differentiable> {
    Ref(&'a Value<'a, T>),
    Const(ValueConst<T>),
}

pub(crate) struct ValueConst<T: Differentiable> {
    data: T,
}

impl<T: Differentiable> ValueConst<T> {
    fn data(&self) -> T {
        self.data
    }
}

impl<'a, T: Differentiable> From<Value<'a, T>> for ValueConst<T> {
    fn from(value: Value<'a, T>) -> Self {
        ValueConst { data: value.data() }
    }
}

struct Add<'a, T: Differentiable> {
    operands: (Operand<'a, T>, Operand<'a, T>)
}

impl<'a, T: Differentiable> Operation<'a, T> for Add<'a, T> {
    fn forward(&mut self, lhs: &'a Value<'a, T>, rhs: &'a Value<'a, T>) -> Value<'a, T> {
        let value = Value::new(lhs.data() + rhs.data());
        self.operands = (Operand::Ref(lhs), Operand::Ref(rhs));
        value
    }
    fn backward(&self, grad: T) {

    }
}
