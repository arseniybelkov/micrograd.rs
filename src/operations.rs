use crate::{Differentiable, Value};
use std::{ops, cell::Cell};

pub trait Operation<'a, T: Differentiable> {
    fn forward(self) -> Value<'a, T>;
    fn backward(&self, grad: T) -> (T, T);
    fn operands(&self) -> (&Operand<'a, T>, &Operand<'a, T>);
}

pub(crate) enum Operand<'a, T: Differentiable> {
    Ref(&'a Value<'a, T>),
    Value(Value<'a, T>),
}

impl<'a, T: Differentiable> Operand<'a, T> {
    pub fn data(&self) -> T {
        match self {
            Self::Ref(v) => v.data(),
            Self::Value(v) => v.data(),
        }
    }

    pub fn value(&self) -> &Value<'a, T> {
        match self {
            Self::Ref(v) => *v,
            Self::Value(v) => v,
        }
    }
}

struct Add<'a, T: Differentiable>(Operand<'a, T>, Operand<'a, T>);

impl<'a, T: Differentiable> Operation<'a, T> for Add<'a, T> {
    fn forward(self) -> Value<'a, T> {
        let (lhs, rhs) = (&self.0, &self.1);
        let mut value = Value::new(lhs.data() + rhs.data());
        value.operation = Cell::new(Some(Box::new(self)));
        value
    }

    fn backward(&self, grad: T) -> (T, T) {
        (grad * T::eye_grad(), grad * T::eye_grad())
    }

    fn operands(&self) -> (&Operand<'a, T>, &Operand<'a, T>) {
        (&self.0, &self.1)
    }
}

impl<'a, T: Differentiable> ops::Add<Value<'a, T>> for Value<'a, T> {
    type Output = Self;
    fn add(self, rhs: Value<'a, T>) -> Self::Output {
        Add(Operand::Value(self), Operand::Value(rhs)).forward()
    }
}

impl<'a, T: Differentiable> ops::Add<Value<'a, T>> for &'a Value<'a, T> {
    type Output = Value<'a, T>;
    fn add(self, rhs: Value<'a, T>) -> Self::Output {
        Add(Operand::Ref(self), Operand::Value(rhs)).forward()
    }
}

impl<'a, T: Differentiable> ops::Add<&'a Value<'a, T>> for Value<'a, T> {
    type Output = Value<'a, T>;
    fn add(self, rhs: &'a Value<'a, T>) -> Self::Output {
        Add(Operand::Value(self), Operand::Ref(rhs)).forward()
    }
}

impl<'a, T: Differentiable> ops::Add<&'a Value<'a, T>> for &'a Value<'a, T> {
    type Output = Value<'a, T>;
    fn add(self, rhs: &'a Value<'a, T>) -> Self::Output {
        Add(Operand::Ref(self), Operand::Ref(rhs)).forward()
    }
}

create_operation! {Add, ops::Add, 1, Value, Value};
create_operation! {Add, ops::Add, 1, Value, Value};
create_operation! {Add, ops::Add, 1, Value, Value};
create_operation! {Add, ops::Add, 1, Value, Value};

struct Neg<'a, T: Differentiable>(Operand<'a, T>);

impl<'a, T: Differentiable> Operation<'a, T> for Neg<'a, T> {
    fn forward(self) -> Value<'a, T> {
        let mut value = Value::new(-self.0.data());
        value.operation = Cell::new(Some(Box::new(self)));
        value
    }

    fn backward(&self, grad: T) -> (T, T) {
        (-T::eye_grad() * grad, -T::eye_grad() * grad)
    }

    fn operands(&self) -> (&Operand<'a, T>, &Operand<'a, T>) {
        (&self.0, &self.0)
    }
}

impl<'a, T: Differentiable> ops::Neg for Value<'a, T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Neg(Operand::Value(self)).forward()
    }
}
