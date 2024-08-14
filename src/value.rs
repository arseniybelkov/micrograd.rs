use crate::Differentiable;
use std::cell::Cell;
use std::ops;
use crate::operations::{Operation, Operand};

pub struct Value<'a, T: Differentiable> {
    data: Cell<T>,
    grad: Option<Cell<T>>,
    pub(super) operation: Cell<Option<Box<dyn Operation<'a, T> + 'a>>>,
}

impl<'a, T: Differentiable> Value<'a, T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Cell::new(data),
            grad: Some(Cell::new(T::zero_grad())),
            operation: Cell::new(None),
        }
    }

    pub fn coeff(data: T) -> Self {
        Self {
            data: Cell::new(data),
            grad: None,
            operation: Cell::new(None),
        }
    }

    pub fn data(&self) -> T {
        self.data.get()
    }

    pub fn grad(&self) -> Option<T> {
        self.grad.as_ref().map(|g| g.get())
    }

    pub fn zero_grad(&self) {
        self.set_grad(T::zero_grad())
    }

    pub(crate) fn set_grad(&self, grad: T) {
        if let Some(ref g) = self.grad {
            g.set(grad);
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
        let Some(operation) = self.operation.take() else {return};
        let (lhs, rhs) = operation.operands();
        let (g1, g2) = operation.backward(grad);
        if std::ptr::eq(lhs, rhs) {
            let value = lhs.value();
            value.set_grad(value.grad().unwrap() + g1);
            value._backward(value.grad().unwrap());
        } else {
            let (lhs, rhs) = (lhs.value(), rhs.value());
            lhs.set_grad(lhs.grad().unwrap() + g1);
            rhs.set_grad(rhs.grad().unwrap() + g2);
            lhs._backward(lhs.grad().unwrap());
            rhs._backward(rhs.grad().unwrap());
        }
    }
}
