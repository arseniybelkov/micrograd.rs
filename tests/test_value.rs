use micrograd::Value;

#[test]
fn simple() {
    let value = Value::new(1f32);
    let result = value * 4.5f32;
    assert_eq!(*result.grad(), 4.5f32);
}