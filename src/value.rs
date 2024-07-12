use std::ops::{Add, Sub, Neg, Mul, Div};

pub struct Value<T> {
    data: T,
    grad: Option<T>,
}

impl<T> Value<T> {
    pub fn new(data: T) -> Self {
        Self { data , grad: None}
    }

    pub fn grad(&self) -> Option<&T> {
        self.grad.as_ref()
    }
}

impl<T> Add<T> for Value<T>
where T: Add<T, Output=T>
{
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        let mut this = Self::new(self.data + rhs);
        this
    }
}

impl<T> Sub<T> for Value<T>
where T: Sub<T, Output=T>
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        let mut this = Self::new(self.data - rhs);
        this
    }
}

impl<T> Mul<T> for Value<T>
where T: Mul<T, Output=T>
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        let mut this = Self::new(self.data * rhs);
        this
    }
}

impl<T> Div<T> for Value<T>
where T: Div<T, Output=T>
{
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        let mut this = Self::new(self.data / rhs);
        this
    }
}

impl<T: Neg<Output=T>> Neg for Value<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let mut this = self;
        this.data = -this.data;
        this
    }
}
