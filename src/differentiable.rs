use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Differentiable:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Sized
{
    fn zero_grad() -> Self;
    fn eye_grad() -> Self;
}

// TODO: macro_rules!

impl Differentiable for f32 {
    fn zero_grad() -> Self {
        0f32
    }

    fn eye_grad() -> Self {
        1f32
    }
}

impl Differentiable for f64 {
    fn zero_grad() -> Self {
        0f64
    }

    fn eye_grad() -> Self {
        1f64
    }
}
