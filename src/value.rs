use crate::differentiable::Differentiable;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

#[derive(Clone)]
enum Operation<T: Differentiable + Copy> {
    Add(Rc<ValueInternal<T>>, Rc<ValueInternal<T>>),
    Sub(Rc<ValueInternal<T>>, Rc<ValueInternal<T>>),
    // Mul(Rc<ValueInternal<T>>, Rc<ValueInternal<T>>),
    // Div(Rc<ValueInternal<T>>, Rc<ValueInternal<T>>),
    // Neg,
}

#[derive(Clone)]
struct ValueInternal<T: Differentiable + Copy> {
    data: Cell<T>,
    grad: Cell<T>,
    operation: Option<Operation<T>>,
}

impl<T: Differentiable + Copy> ValueInternal<T> {
    fn backward(&self) {
        todo!()
    }
}

#[derive(Clone)]
pub struct Value<T: Differentiable + Copy>(Rc<ValueInternal<T>>);

impl<T: Differentiable + Copy> Value<T> {
    pub fn new(data: T) -> Self {
        let inner = ValueInternal {
            data: Cell::new(data),
            grad: Cell::new(T::zero_grad()),
            operation: None,
        };
        Self(Rc::new(inner))
    }

    fn from_internal(internal: ValueInternal<T>) -> Self {
        Self(Rc::new(internal))
    }

    pub fn grad(&self) -> T {
        self.0.grad.get()
    }

    pub fn zero_grad(&mut self) {
        self.0.grad.set(T::zero_grad());
    }

    pub fn backward(&self) {
        match self.0.operation {
            Some(op) => match op {
                Operation::Add(lhs, rhs) => {
                    lhs.backward();
                    rhs.backward();
                },
                Operation::Sub(lhs, rhs) => {
                    lhs.backward();
                    rhs.backward();
                },
            },
            None => {},
        }
    }
}

// impl<T> Add<T> for &Value<T>
// where
//     T: Add<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T>;
//     fn add(self, rhs: T) -> Self::Output {
//         let mut value = Value::new(self.0.data.get() + rhs);
//         value.0.operation = Some(Operation::Add(self.0.clone(), rhs));
//         value
//     }
// }

// impl<T> Add<Value<T>> for T
// where
//     T: Add<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T>;
//     fn add(self, rhs: Value<T>) -> Self::Output {
//         todo!()
//     }
// }

impl<T> Add<&Value<T>> for &Value<T>
where
    T: Add<T, Output = T> + Differentiable + Copy,
{
    type Output = Value<T>;
    fn add(self, rhs: &Value<T>) -> Self::Output {
        let internal = ValueInternal {
            data: Cell::new(self.0.data.get() + rhs.0.data.get()),
            grad: Cell::new(T::zero_grad()),
            operation: Some(Operation::Add(self.0.clone(), rhs.0.clone())),
        };
        Value::from_internal(internal)
    }
}

// impl<T> Sub<T> for &Value<T>
// where
//     T: Sub<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T, Rc<Cell<T>>, T>;
//     fn sub(self, rhs: T) -> Self::Output {
//         let mut value = Value::new(self.data.get() - rhs);
//         value.operation = Some(Operation::Sub(self.data.clone(), rhs));
//         value
//     }
// }

impl<T> Sub<&Value<T>> for &Value<T>
where
    T: Sub<T, Output = T> + Differentiable + Copy,
{
    type Output = Value<T>;
    fn sub(self, rhs: &Value<T>) -> Self::Output {
        let internal = ValueInternal {
            data: Cell::new(self.0.data.get() - rhs.0.data.get()),
            grad: Cell::new(T::zero_grad()),
            operation: Some(Operation::Sub(self.0.clone(), rhs.0.clone())),
        };
        Value::from_internal(internal)
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

// impl<T> Mul<&Value<T>> for &Value<T>
// where
//     T: Mul<T, Output = T> + Differentiable + Copy,
// {
//     type Output = Value<T>;
//     fn mul(self, rhs: &Value<T>) -> Self::Output {
//         let mut value = Value::new(self.data.get() * rhs.data.get());
//         value.operation = Some(Operation::Mul(self.data.clone(), rhs.data.clone()));
//         value
//     }
// }

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
