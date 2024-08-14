use crate::impl_binary_operation;
use crate::{Differentiable, Value};
use std::{cell::Cell, ops};

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

struct Sub<'a, T: Differentiable>(Operand<'a, T>, Operand<'a, T>);

impl<'a, T: Differentiable> Operation<'a, T> for Sub<'a, T> {
    fn forward(self) -> Value<'a, T> {
        let (lhs, rhs) = (&self.0, &self.1);
        let mut value = Value::new(lhs.data() - rhs.data());
        value.operation = Cell::new(Some(Box::new(self)));
        value
    }

    fn backward(&self, grad: T) -> (T, T) {
        (grad * T::eye_grad(), -grad * T::eye_grad())
    }

    fn operands(&self) -> (&Operand<'a, T>, &Operand<'a, T>) {
        (&self.0, &self.1)
    }
}

struct Mul<'a, T: Differentiable>(Operand<'a, T>, Operand<'a, T>);

impl<'a, T: Differentiable> Operation<'a, T> for Mul<'a, T> {
    fn forward(self) -> Value<'a, T> {
        let (lhs, rhs) = (&self.0, &self.1);
        let mut value = Value::new(lhs.data() * rhs.data());
        value.operation = Cell::new(Some(Box::new(self)));
        value
    }

    fn backward(&self, grad: T) -> (T, T) {
        let (lhs, rhs) = self.operands();
        (grad * rhs.data(), grad * lhs.data())
    }

    fn operands(&self) -> (&Operand<'a, T>, &Operand<'a, T>) {
        (&self.0, &self.1)
    }
}

struct Div<'a, T: Differentiable>(Operand<'a, T>, Operand<'a, T>);

impl<'a, T: Differentiable> Operation<'a, T> for Div<'a, T> {
    fn forward(self) -> Value<'a, T> {
        let (lhs, rhs) = (&self.0, &self.1);
        let mut value = Value::new(lhs.data() / rhs.data());
        value.operation = Cell::new(Some(Box::new(self)));
        value
    }

    fn backward(&self, grad: T) -> (T, T) {
        let (lhs, rhs) = self.operands();
        (
            grad / rhs.data(),
            -grad * lhs.data() / (rhs.data() * rhs.data()),
        )
    }

    fn operands(&self) -> (&Operand<'a, T>, &Operand<'a, T>) {
        (&self.0, &self.1)
    }
}

impl_binary_operation! {Add, std::ops::Add<Value<'a, T>>, std::ops::Add<&'a Value<'a, T>>, add}
impl_binary_operation! {Sub, std::ops::Sub<Value<'a, T>>, std::ops::Sub<&'a Value<'a, T>>, sub}
impl_binary_operation! {Mul, std::ops::Mul<Value<'a, T>>, std::ops::Mul<&'a Value<'a, T>>, mul}
impl_binary_operation! {Div, std::ops::Div<Value<'a, T>>, std::ops::Div<&'a Value<'a, T>>, div}

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
