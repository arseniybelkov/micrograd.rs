use micrograd::Value;
use pretty_assertions::assert_eq;

#[test]
fn test_simple() {
    let value = Value::new(1f32);
    let x = Value::new(3f32);
    let result = &value + &x;
    result.backward();
    assert_eq!(value.grad().unwrap(), 1f32);

    let w1 = Value::new(5f32);
    let w2 = Value::new(4f32);
    let w3 = Value::new(3f32);

    let w = &w1 + &w2;
    let result = &w * &w3;
    result.backward();

    assert_eq!(w1.grad().unwrap(), 3f32);
    assert_eq!(w2.grad().unwrap(), 3f32);
    assert_eq!(w3.grad().unwrap(), 9f32);
}

// #[test]
// fn test_sigmoid() {
//     fn sigmoid<'a>(x: &'a Value<'a, f32>) -> Value<'a, f32> {
//         Value::coeff(1f32) / (Value::coeff(1f32))
//     }
// }

#[test]
fn test_deep() {
    let x = Value::new(5f64);
    let y = Value::new(4f64);
    let z = Value::new(8f64);

    let a = &y + &x;
    let b = &z * &a;
    let c = &b * &b;
    let d = &c - &z;
    let e = &z * &d;
    let f = &e + &e;
    let result = &f / &x;

    result.backward();
    assert!(f64::abs(x.grad().unwrap() - 373.76) < 10e-8f64);
    assert!(f64::abs(y.grad().unwrap() - 3686.4) < 10e-8f64);
    assert!(f64::abs(z.grad().unwrap() - 6214.4) < 10e-8f64);
}

#[cfg(test)]
mod operations {
    use super::{assert_eq, Value};

    #[test]
    fn test_add() {
        let x = Value::new(1f32);
        let y = Value::new(2f32);
        (&x + &y).backward();
        assert_eq!(x.grad().unwrap(), 1f32);
        assert_eq!(y.grad().unwrap(), 1f32);

        let z = Value::new(5f32);
        (&x + &z).backward();
        assert_eq!(x.grad().unwrap(), 2f32);
        assert_eq!(y.grad().unwrap(), 1f32);
        assert_eq!(z.grad().unwrap(), 1f32);

        x.zero_grad();
        y.zero_grad();

        let z = &x + &y;
        (&z + &x).backward();

        assert_eq!(x.grad().unwrap(), 2f32);

        let x = Value::new(3f32);
        (&x + &x).backward();
        assert_eq!(x.grad().unwrap(), 2f32);
    }

    #[test]
    fn test_sub() {
        let x = Value::new(1f32);
        let y = Value::new(2f32);
        (&x - &y).backward();
        assert_eq!(x.grad().unwrap(), 1f32);
        assert_eq!(y.grad().unwrap(), -1f32);

        let z = Value::new(5f32);
        (&x - &z).backward();
        assert_eq!(x.grad().unwrap(), 2f32);
        assert_eq!(y.grad().unwrap(), -1f32);
        assert_eq!(z.grad().unwrap(), -1f32);

        x.zero_grad();
        y.zero_grad();

        let x = Value::new(13f32);
        let z = &x - &y;
        (&z - &x).backward();

        assert_eq!(x.grad().unwrap(), 0f32);
    }

    #[test]
    fn test_mul() {
        let x = Value::new(1f32);
        let y = Value::new(2f32);
        (&x * &y).backward();
        assert_eq!(x.grad().unwrap(), 2f32);
        assert_eq!(y.grad().unwrap(), 1f32);

        let z = Value::new(5f32);
        (&x * &z).backward();
        assert_eq!(x.grad().unwrap(), 7f32);
        assert_eq!(y.grad().unwrap(), 1f32);
        assert_eq!(z.grad().unwrap(), 1f32);

        x.zero_grad();
        y.zero_grad();

        let x = Value::new(13f32);
        let z = &x * &y;
        (&z * &x).backward();

        assert_eq!(x.grad().unwrap(), 52f32);

        let x = Value::new(4f32);
        (&x * &x).backward();
        assert_eq!(x.grad().unwrap(), 8f32);
    }

    #[test]
    fn test_div() {
        let x = Value::new(3f32);
        let y = Value::new(2f32);
        (&x / &y).backward();
        assert_eq!(x.grad().unwrap(), 0.5);
        assert_eq!(y.grad().unwrap(), -0.75);

        let z = Value::new(5f32);
        (&x / &z).backward();
        assert_eq!(x.grad().unwrap(), 0.7);
        assert_eq!(y.grad().unwrap(), -0.75);
        assert_eq!(z.grad().unwrap(), -0.12);

        x.zero_grad();
        y.zero_grad();

        let x = Value::new(13f32);
        let z = &x / &y;
        (&z / &x).backward();

        assert_eq!(x.grad().unwrap(), 0f32);

        let x = Value::new(4f32);
        (&x / &x).backward();
        assert_eq!(x.grad().unwrap(), 0f32);
    }

    #[test]
    fn test_pow() {
        let x = Value::new(1f32);
        let y = Value::new(2f32);

        let result = x.pow(&y);
        result.backward();
        assert_eq!(x.grad().unwrap(), 2f32);
        assert_eq!(y.grad().unwrap(), 0f32);

        let x = Value::new(2f64);
        let y = Value::new(3f64);
        let z = Value::new(0.1f64);

        let a = x.pow(&y);
        let b = &a + &z;
        let result = b.pow(&z);

        result.backward();
        assert!(f64::abs(x.grad().unwrap() - 0.1826f64) < 1e-3f64);
        assert!(f64::abs(y.grad().unwrap() - 0.0844f64) < 1e-3f64);
        assert!(f64::abs(z.grad().unwrap() - 2.5938f64) < 1e-3f64);

        let x = Value::new(3f64);
        x.pow(&x).backward();
        assert!(f64::abs(x.grad().unwrap() - 56.6625f64) < 1e-3f64);
    }

    #[test]
    fn test_neg() {
        let x = Value::new(123f32);
        let y = -&x;
        y.backward();
        assert_eq!(x.grad().unwrap(), -1f32);
    }
}
