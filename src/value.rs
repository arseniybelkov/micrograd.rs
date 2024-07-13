use crate::differentiable::Differentiable;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone)]
pub struct Value<T: Differentiable> {
    data: T,
    grad: T,
}

impl<T: Differentiable> Value<T> {
    pub fn new(data: T) -> Self {
        Self {
            data,
            grad: T::zero_grad(),
        }
    }

    pub fn grad(&self) -> &T {
        &self.grad
    }

    pub fn zero_grad(&mut self) {
        self.grad = T::zero_grad();
    }
}

impl<T, R, O> Add<R> for Value<T>
where
    T: Add<R, Output = O> + Differentiable,
    O: Differentiable,
{
    type Output = Value<O>;
    fn add(self, rhs: R) -> Self::Output {
        Value::new(self.data + rhs)
    }
}

impl<T, R, O> Sub<R> for Value<T>
where
    T: Sub<R, Output = O> + Differentiable,
    O: Differentiable,
{
    type Output = Value<O>;
    fn sub(self, rhs: R) -> Self::Output {
        Value::new(self.data - rhs)
    }
}

impl<T, R, O> Mul<R> for Value<T>
where
    T: Mul<R, Output = O> + Differentiable + Clone,
    R: Clone,
    O: Differentiable,
{
    type Output = Value<O>;
    fn mul(self, rhs: R) -> Self::Output {
        let data: O = self.data * rhs.clone();
        let grad: O = self.grad * rhs;
        let mut value = Value::new(data);
        value.grad = grad;
        value
    }
}

impl<T, R, O> Div<R> for Value<T>
where
    T: Div<R, Output = O> + Differentiable + Clone,
    R: Clone,
    O: Differentiable,
{
    type Output = Value<O>;
    fn div(mut self, rhs: R) -> Self::Output {
        let data = self.data / rhs.clone();
        let grad = self.grad / rhs;
        let mut value = Value::new(data);
        value.grad = grad;
        value
    }
}

impl<T> Neg for Value<T>
where
    T: Neg<Output = T> + Differentiable,
{
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        self.data = -self.data;
        self.grad = -self.grad;
        self
    }
}
