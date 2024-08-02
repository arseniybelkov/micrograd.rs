use std::cell::Cell;
use crate::{Differentiable, Value};

pub trait Operation<'a, T: Differentiable, const N: usize = 2> {
    fn forward(&mut self, operands: [Operand<'a, T>; N]) -> Value<'a, T>;
    fn backward(&self, grad: T);
}

pub struct Operand<'a, T: Differentiable> {
    
}