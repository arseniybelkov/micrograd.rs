use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Differentiable:
    // TODO: should Differentiable require Float
    Float
    + Add<Output = Self>
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

pub trait Float: Copy {
    fn pow(self, n: Self) -> Self;
    fn log(self) -> Self;
}

impl Float for f32 {
    fn pow(self, n: Self) -> Self {
        self.powf(n)
    }

    fn log(self) -> Self {
        self.ln()
    }
}

impl Float for f64 {
    fn pow(self, n: Self) -> Self {
        self.powf(n)
    }

    fn log(self) -> Self {
        self.ln()
    }
}
