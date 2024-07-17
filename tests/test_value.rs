use pretty_assertions::assert_eq;
use micrograd::Value;

#[test]
fn test_simple() {
    let value = Value::new(1f32);
    let x = Value::new(3f32);
    let result = &value + &x;
    result.backward();
    assert_eq!(value.grad(), 1f32);

    let w1 = Value::new(5f32);
    let w2 = Value::new(4f32);
    let w3 = Value::new(3f32);

    let w = &w1 + &w2;
    let result =  &w * &w3;
    result.backward();

    assert_eq!(w1.grad(), 3f32);
    assert_eq!(w2.grad(), 3f32);
    assert_eq!(w3.grad(), 9f32);
}

#[cfg(test)]
mod operations {
    use super::{Value, assert_eq};

    #[test]
    fn test_add() {
        let x = Value::new(1f32);
        let y = Value::new(2f32);
        (&x + &y).backward();
        assert_eq!(x.grad(), 1f32);
        assert_eq!(y.grad(), 1f32);

        let z = Value::new(5f32);
        (&x + &z).backward();
        assert_eq!(x.grad(), 2f32);
        assert_eq!(y.grad(), 1f32);
        assert_eq!(z.grad(), 1f32);

        x.zero_grad();
        y.zero_grad();

        let z = &x + &y;
        (&z + &x).backward();

        assert_eq!(x.grad(), 2f32);

        let x = Value::new(3f32);
        (&x + &x).backward();
        assert_eq!(x.grad(), 2f32);
    }

    #[test]
    fn test_mul() {
        let x = Value::new(1f32);
        let y = Value::new(2f32);
        (&x * &y).backward();
        assert_eq!(x.grad(), 2f32);
        assert_eq!(y.grad(), 1f32);

        let z = Value::new(5f32);
        (&x * &z).backward();
        assert_eq!(x.grad(), 7f32);
        assert_eq!(y.grad(), 1f32);
        assert_eq!(z.grad(), 1f32);

        x.zero_grad();
        y.zero_grad();

        let x = Value::new(13f32);
        let z = &x * &y;
        (&z * &x).backward();

        assert_eq!(x.grad(), 52f32);
    }
}
