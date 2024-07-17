use crate::differentiable::Differentiable;
use std::cell::Cell;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone)]
enum Operation<'a, T: Differentiable + Copy> {
    Add(&'a Value<'a, T>, &'a Value<'a, T>),
    Sub(&'a Value<'a, T>, &'a Value<'a, T>),
    Mul(&'a Value<'a, T>, &'a Value<'a, T>),
    // Div(Rc<ValueInternal<T>>, Rc<ValueInternal<T>>),
    // Neg,
}

#[derive(Clone)]
pub struct Value<'a, T: Differentiable + Copy> {
    data: Cell<T>,
    grad: Cell<T>,
    operation: Option<Operation<'a, T>>,
}

impl<'a, T: Differentiable + Copy> Value<'a, T> {
    pub fn new(data: T) -> Self {
        Self {
            data: Cell::new(data),
            grad: Cell::new(T::zero_grad()),
            operation: None,
        }
    }

    pub fn grad(&self) -> T {
        self.grad.get()
    }

    pub fn zero_grad(&self) {
        self.grad.set(T::zero_grad());
    }

    pub fn backward(&self) {
        // dy / dy = 1
        self._backward(T::eye_grad());
    }

    fn _backward(&self, grad: T) {
        match &self.operation {
            Some(op) => {
                let (v1, v2) = match op {
                    Operation::Add(v1, v2) => {
                        v1.grad.set(v1.grad() + T::eye_grad() * grad);
                        v2.grad.set(v2.grad() + T::eye_grad() * grad);
                        (v1, v2)
                    }
                    Operation::Mul(v1, v2) => {
                        v1.grad.set(v1.grad() + v2.data.get() * grad);
                        v2.grad.set(v2.grad() + v1.data.get() * grad);
                        (v1, v2)
                    }
                    Operation::Sub(v1, v2) => {
                        v1.grad.set(v1.grad() + T::eye_grad() * grad);
                        v2.grad.set(v2.grad() - T::eye_grad() * grad);
                        (v1, v2)
                    }
                };

                v1._backward(v1.grad());
                v2._backward(v2.grad());
            }
            None => {
                // end of graph
            }
        }
    }
}

impl<'a, T> Add<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Add<T, Output = T> + Differentiable + Copy,
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
    T: Sub<T, Output = T> + Differentiable + Copy,
{
    type Output = Value<'a, T>;
    fn sub(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data.get() - rhs.data.get());
        value.operation = Some(Operation::Sub(self, rhs));
        value
    }
}

// impl<T> Sub<Value<T>> for T
// where
//     T: Sub<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T>;
//     fn sub(self, rhs: Value<T>) -> Self::Output {
//         let data = self - rhs.data;
//         let grad = T::zero_grad() - rhs.grad;
//         let mut value = Value::new(data);
//         value.grad = grad;
//         value
//     }
// }

// impl<T> Mul<T> for &Value<T>
// where
//     T: Mul<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T, Rc<Cell<T>>, T>;
//     fn mul(self, rhs: T) -> Self::Output {
//         let mut value = Value::new(self.data.get() * rhs);
//         value.operation = Some(Operation::Mul(self.data.clone(), rhs));
//         value
//     }
// }

impl<'a, T> Mul<&'a Value<'a, T>> for &'a Value<'a, T>
where
    T: Mul<T, Output = T> + Differentiable + Copy,
{
    type Output = Value<'a, T>;
    fn mul(self, rhs: &'a Value<'a, T>) -> Self::Output {
        let mut value = Value::new(self.data.get() * rhs.data.get());
        value.operation = Some(Operation::Mul(self, rhs));
        value
    }
}

// impl<T> Div<T> for &Value<T>
// where
//     T: Div<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T, Rc<Cell<T>>, T>;
//     fn div(self, rhs: T) -> Self::Output {
//         let mut value = Value::new(self.data.get() / rhs);
//         value.operation = Some(Operation::Div(self.data.clone(), rhs));
//         value
//     }
// }

// impl<T> Div<&Value<T>> for &Value<T>
// where
//     T: Div<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T>;
//     fn div(self, rhs: &Value<T>) -> Self::Output {
//         let mut value = Value::new(self.data.get() / self.data.get());
//         value.operation = Some(Operation::Div(self.data.clone(), rhs.data.clone()));
//         value
//     }
// }

// impl<T> Neg for &Value<T>
// where
//     T: Neg<Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T>;
//     fn neg(self) -> Self::Output {
//         let mut value = Value::new(-self.data.get());
//         value.operation = Some(Operation::Neg(self.data.clone()));
//         value
//     }
// }
