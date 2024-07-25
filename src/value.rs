use crate::Differentiable;
use crate::unwrap_operands;
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

enum Operand<'a, T: Differentiable> {
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
}

struct ValueConst<T: Differentiable> {
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
    grad: Cell<T>,
    operation: Option<Operation<'a, T>>,
    pub requires_grad: bool,
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
            grad: Cell::new(T::zero_grad()),
            requires_grad: true,
            operation: None,
        }
    }

    pub fn coeff(data: T) -> Self {
        Self {
            data: Cell::new(data),
            grad: Cell::new(T::zero_grad()),
            requires_grad: false,
            operation: None,
        }
    }

    pub fn data(&self) -> T {
        self.data.get()
    }

    pub fn grad(&self) -> T {
        self.grad.get()
    }

    pub fn zero_grad(&self) {
        self.grad.set(T::zero_grad())
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
                    unwrap_operands!(
                        v1,
                        v1.grad.set(v1.grad() + T::eye_grad() * grad),
                        v2,
                        v2.grad.set(v2.grad() + T::eye_grad() * grad)
                    );
                }
                Operation::Mul(v1, v2) => {
                    unwrap_operands!(
                        v1,
                        v1.grad.set(v1.grad() + v2.data() * grad),
                        v2,
                        v2.grad.set(v2.grad() + v1.data() * grad)
                    );
                }
                Operation::Sub(v1, v2) => {
                    unwrap_operands!(
                        v1,
                        v1.grad.set(v1.grad() + T::eye_grad() * grad),
                        v2,
                        v2.grad.set(v2.grad() - T::eye_grad() * grad)
                    );
                }
                Operation::Div(v1, v2) => {
                    unwrap_operands!(
                        v1,
                        v1.grad.set(v1.grad() + T::eye_grad() / v2.data() * grad),
                        v2,
                        v2.grad
                            .set(v2.grad() - v1.data() / (v2.data() * v2.data()) * grad)
                    );
                }
                Operation::Neg(v) => {
                    if let Operand::Ref(v) = v {
                        v.grad.set(v.grad() - T::eye_grad() * grad);
                        v._backward(v.grad());
                    }
                }
                Operation::Pow(v1, v2) => {
                    unwrap_operands!(
                        v1,
                        v1.grad.set(
                            v1.grad() + v2.data() * v1.data().pow(v2.data() - T::eye_grad()) * grad
                        ),
                        v2,
                        v2.grad
                            .set(v2.grad() + v1.data().pow(v2.data()) * v1.data().log() * grad)
                    );
                }
            },
            None => {
                // end of graph
            }
        }
    }
}

fn pair_backward<'a, T: Differentiable>(v1: &'a Value<T>, v2: &'a Value<T>) {
    v1._backward(v1.grad());
    if !std::ptr::eq(v1, v2) {
        v2._backward(v2.grad());
    }
}

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
    T: Differentiable
{
    type Output = Value<'a, T>;
    fn add(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() + rhs.data());
        value.operation = Some(Operation::Add(Operand::Const(self.into()), Operand::Ref(rhs)));
        value
    }
}

impl<'a, T> Add<Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable
{
    type Output = Value<'a, T>;
    fn add(self, rhs: Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data() + rhs.data());
        value.operation = Some(Operation::Add(Operand::Ref(self), Operand::Const(rhs.into())));
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
        value.operation = Some(Operation::Sub(Operand::Const(self.into()), Operand::Ref(rhs)));
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
        value.operation = Some(Operation::Sub(Operand::Ref(self), Operand::Const(rhs.into())));
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
        value.operation = Some(Operation::Mul(Operand::Const(self.into()), Operand::Ref(rhs)));
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
        value.operation = Some(Operation::Mul(Operand::Ref(self), Operand::Const(rhs.into())));
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
        value.operation = Some(Operation::Div(Operand::Const(self.into()), Operand::Ref(rhs)));
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
        value.operation = Some(Operation::Div(Operand::Ref(self), Operand::Const(rhs.into())));
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
