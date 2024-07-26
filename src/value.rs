use crate::backward;
use crate::Differentiable;
use std::cell::Cell;
use std::ops::{Add, Div, Mul, Neg, Sub};

enum Operation<'a, T: Differentiable> {
    Add(Operand<'a, T>, Operand<'a, T>),
    Sub(Operand<'a, T>, Operand<'a, T>),
    Mul(Operand<'a, T>, Operand<'a, T>),
    Div(Operand<'a, T>, Operand<'a, T>),
    Pow(Operand<'a, T>, Operand<'a, T>),
    Neg(Operand<'a, T>),
}

pub(crate) enum Operand<'a, T: Differentiable> {
    Ref(&'a Value<'a, T>),
    Const(ValueConst<T>),
}

impl<'a, T: Differentiable> Operand<'a, T> {
    fn data(&self) -> T {
        match self {
            Operand::Ref(v) => v.data(),
            Operand::Const(v) => v.data(),
        }
    }

    fn set_grad(&self, value: T) {
        if let Self::Ref(s) = self {
            if let Some(ref grad) = s.grad {
                grad.set(s.grad().unwrap() + value);
            }
        }
    }
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

pub struct Value<'a, T: Differentiable> {
    data: Cell<T>,
    grad: Option<Cell<T>>,
    operation: Option<Operation<'a, T>>,
}

impl<'a, T: Differentiable> Value<'a, T> {
    pub fn pow(&'a self, n: &'a Value<'a, T>) -> Value<'a, T> {
        let mut value = Self::new(self.data().pow(n.data.get()));
        value.operation = Some(Operation::Pow(Operand::Ref(self), Operand::Ref(n)));
        value
    }
}

impl<'a, T: Differentiable> Value<'a, T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Cell::new(data),
            grad: Some(Cell::new(T::zero_grad())),
            operation: None,
        }
    }

    pub fn coeff(data: T) -> Self {
        Self {
            data: Cell::new(data),
            grad: None,
            operation: None,
        }
    }

    pub fn data(&self) -> T {
        self.data.get()
    }

    pub fn grad(&self) -> Option<T> {
        self.grad.as_ref().map(|g| g.get())
    }

    pub fn zero_grad(&self) {
        if let Some(ref g) = self.grad {
            g.set(T::zero_grad())
        }
    }

    pub fn requires_grad(&mut self, val: bool) {
        if val && self.grad.is_none() {
            self.grad = Some(Cell::new(T::zero_grad()));
        } else if !val {
            self.grad = None;
        }
    }
}

impl<'a, T: Differentiable> Value<'a, T> {
    pub fn backward(&self) {
        // dy / dy = 1
        self._backward(T::eye_grad());
    }

    fn _backward(&self, grad: T) {
        match &self.operation {
            Some(op) => match op {
                Operation::Add(v1, v2) => {
                    v1.set_grad(T::eye_grad() * grad);
                    v2.set_grad(T::eye_grad() * grad);
                    backward!(v1, v2);
                }
                Operation::Mul(v1, v2) => {
                    v1.set_grad(v2.data() * grad);
                    v2.set_grad(v1.data() * grad);
                    backward!(v1, v2);
                }
                Operation::Sub(v1, v2) => {
                    v1.set_grad(T::eye_grad() * grad);
                    v2.set_grad(-T::eye_grad() * grad);
                    backward!(v1, v2);
                }
                Operation::Div(v1, v2) => {
                    v1.set_grad(T::eye_grad() / v2.data() * grad);
                    v2.set_grad(-v1.data() / (v2.data() * v2.data()) * grad);
                    backward!(v1, v2);
                }
                Operation::Neg(v) => {
                    v.set_grad(-T::eye_grad() * grad);
                    backward!(v);
                }
                Operation::Pow(v1, v2) => {
                    v1.set_grad(v2.data() * v1.data().pow(v2.data() - T::eye_grad()) * grad);
                    v2.set_grad(v1.data().pow(v2.data()) * v1.data().log() * grad);
                    backward!(v1, v2);
                }
            },
            None => {
                // end of graph
            }
        }
    }
}

// fn pair_backward<'a, T: Differentiable>(v1: &'a Value<T>, v2: &'a Value<T>) {
//     match (&v1.grad, &v2.grad) {
//         (Some(_), Some(_)) => {
//             v1._backward(v1.grad().unwrap());
//             if !std::ptr::eq(v1, v2) {
//                 v2._backward(v2.grad().unwrap());
//             }
//         },
//         (Some(_), None) => v1._backward(v1.grad().unwrap()),
//         (None, Some(_)) => v2._backward(v2.grad().unwrap()),
//         (None, None) => {},
//     }
// }

// TODO: stuff it all into macros

impl<'a, T> Add<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn add(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() + rhs.data());
        value.operation = Some(Operation::Add(Operand::Ref(self), Operand::Ref(rhs)));
        value
    }
}

impl<'a, T> Add<&'a Value<'a, T>> for Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn add(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() + rhs.data());
        value.operation = Some(Operation::Add(
            Operand::Const(self.into()),
            Operand::Ref(rhs),
        ));
        value
    }
}

impl<'a, T> Add<Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn add(self, rhs: Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() + rhs.data());
        value.operation = Some(Operation::Add(
            Operand::Ref(self),
            Operand::Const(rhs.into()),
        ));
        value
    }
}

impl<'a, T> Sub<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn sub(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() - rhs.data());
        value.operation = Some(Operation::Sub(Operand::Ref(self), Operand::Ref(rhs)));
        value
    }
}

impl<'a, T> Sub<&'a Value<'a, T>> for Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn sub(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() - rhs.data());
        value.operation = Some(Operation::Sub(
            Operand::Const(self.into()),
            Operand::Ref(rhs),
        ));
        value
    }
}

impl<'a, T> Sub<Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn sub(self, rhs: Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() - rhs.data());
        value.operation = Some(Operation::Sub(
            Operand::Ref(self),
            Operand::Const(rhs.into()),
        ));
        value
    }
}

impl<'a, T> Mul<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn mul(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() * rhs.data());
        value.operation = Some(Operation::Mul(Operand::Ref(self), Operand::Ref(rhs)));
        value
    }
}

impl<'a, T> Mul<&'a Value<'a, T>> for Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn mul(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() * rhs.data());
        value.operation = Some(Operation::Mul(
            Operand::Const(self.into()),
            Operand::Ref(rhs),
        ));
        value
    }
}

impl<'a, T> Mul<Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn mul(self, rhs: Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() * rhs.data());
        value.operation = Some(Operation::Mul(
            Operand::Ref(self),
            Operand::Const(rhs.into()),
        ));
        value
    }
}

impl<'a, T> Div<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn div(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() / rhs.data());
        value.operation = Some(Operation::Div(Operand::Ref(self), Operand::Ref(rhs)));
        value
    }
}

impl<'a, T> Div<&'a Value<'a, T>> for Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn div(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() / rhs.data());
        value.operation = Some(Operation::Div(
            Operand::Const(self.into()),
            Operand::Ref(rhs),
        ));
        value
    }
}

impl<'a, T> Div<Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn div(self, rhs: Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() / rhs.data());
        value.operation = Some(Operation::Div(
            Operand::Ref(self),
            Operand::Const(rhs.into()),
        ));
        value
    }
}

impl<'a, T> Neg for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn neg(self) -> Self::Output {
        let mut value = Value::new(-self.data());
        value.operation = Some(Operation::Neg(Operand::Ref(self)));
        value
    }
}

impl<'a, T> Neg for Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn neg(self) -> Self::Output {
        let mut value = Value::new(-self.data());
        value.operation = Some(Operation::Neg(Operand::Const(self.into())));
        value
    }
}
