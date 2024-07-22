use crate::Differentiable;
use std::cell::Cell;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone)]
enum Operation<'a, T: Copy> {
    Add(&'a Value<'a, T>, &'a Value<'a, T>),
    Sub(&'a Value<'a, T>, &'a Value<'a, T>),
    Mul(&'a Value<'a, T>, &'a Value<'a, T>),
    Div(&'a Value<'a, T>, &'a Value<'a, T>),
    Pow(&'a Value<'a, T>, &'a Value<'a, T>),
    Neg(&'a Value<'a, T>),
}

#[derive(Clone)]
pub struct Value<'a, T: Copy> {
    data: Cell<T>,
    grad: Cell<T>,
    operation: Option<Operation<'a, T>>,
}

impl<'a, T: Differentiable> Value<'a, T> {
    pub fn pow(&'a self, n: &'a Value<'a, T>) -> Value<'a, T> {
        let mut value = Self::new(self.data.get().pow(n.data.get()));
        value.operation = Some(Operation::Pow(self, n));
        value
    }
}

impl<'a, T: Differentiable + Copy> Value<'a, T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Cell::new(data),
            grad: Cell::new(T::zero_grad()),
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
        self.grad.set(T::zero_grad());
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
                    v1.grad.set(v1.grad() + T::eye_grad() * grad);
                    v2.grad.set(v2.grad() + T::eye_grad() * grad);
                    pair_backward(v1, v2)
                }
                Operation::Mul(v1, v2) => {
                    v1.grad.set(v1.grad() + v2.data.get() * grad);
                    v2.grad.set(v2.grad() + v1.data.get() * grad);
                    pair_backward(v1, v2)
                }
                Operation::Sub(v1, v2) => {
                    v1.grad.set(v1.grad() + T::eye_grad() * grad);
                    v2.grad.set(v2.grad() - T::eye_grad() * grad);
                    pair_backward(v1, v2)
                }
                Operation::Div(v1, v2) => {
                    let d2 = v2.data.get();
                    v1.grad.set(v1.grad() + T::eye_grad() / d2 * grad);
                    v2.grad.set(v2.grad() - v1.data.get() / (d2 * d2) * grad);
                    pair_backward(v1, v2)
                }
                Operation::Neg(v) => {
                    v.grad.set(v.grad() - T::eye_grad() * grad);
                    v._backward(v.grad());
                }
                Operation::Pow(v1, v2) => {
                    v1.grad.set(
                        v1.grad()
                            + v2.data.get()
                                * v1.data.get().pow(v2.data.get() - T::eye_grad())
                                * grad,
                    );
                    v2.grad.set(
                        v2.grad() + v1.data.get().pow(v2.data.get()) * v1.data.get().log() * grad,
                    );
                    pair_backward(v1, v2)
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

impl<'a, T> Add<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn add(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data.get() + rhs.data.get());
        value.operation = Some(Operation::Add(self, rhs));
        value
    }
}

impl<'a, T> Sub<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn sub(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data.get() - rhs.data.get());
        value.operation = Some(Operation::Sub(self, rhs));
        value
    }
}

impl<'a, T> Mul<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn mul(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data.get() * rhs.data.get());
        value.operation = Some(Operation::Mul(self, rhs));
        value
    }
}

impl<'a, T> Div<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn div(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data.get() / rhs.data.get());
        value.operation = Some(Operation::Div(self, rhs));
        value
    }
}

impl<'a, T> Neg for &'a Value<'a, T>
where
    T: Differentiable,
{
    type Output = Value<'a, T>;
    fn neg(self) -> Self::Output {
        let mut value = Value::new(-self.data.get());
        value.operation = Some(Operation::Neg(self));
        value
    }
}
