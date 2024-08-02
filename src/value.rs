use crate::backward;
use crate::Differentiable;
use std::cell::Cell;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub struct Value<'a, T: Differentiable> {
    data: Cell<T>,
    pub(crate) grad: Option<Cell<T>>,
    // operation: Option<Operation<'a, T>>,
    _marker: std::marker::PhantomData<&'a T>,
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
