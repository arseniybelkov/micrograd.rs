#[macro_export]
macro_rules! impl_binary_operation {
    ($op:ident, $trt_value:ty, $trt_ref:ty, $method:ident) => {
        impl<'a, T: Differentiable> $trt_value for Value<'a, T> {
            type Output = Value<'a, T>;
            fn $method(self, rhs: Value<'a, T>) -> Self::Output {
                $op(Operand::Value(self), Operand::Value(rhs)).forward()
            }
        }

        impl<'a, T: Differentiable> $trt_ref for Value<'a, T> {
            type Output = Value<'a, T>;
            fn $method(self, rhs: &'a Value<'a, T>) -> Self::Output {
                $op(Operand::Value(self), Operand::Ref(rhs)).forward()
            }
        }

        impl<'a, T: Differentiable> $trt_value for &'a Value<'a, T> {
            type Output = Value<'a, T>;
            fn $method(self, rhs: Value<'a, T>) -> Self::Output {
                $op(Operand::Ref(self), Operand::Value(rhs)).forward()
            }
        }

        impl<'a, T: Differentiable> $trt_ref for &'a Value<'a, T> {
            type Output = Value<'a, T>;
            fn $method(self, rhs: &'a Value<'a, T>) -> Self::Output {
                $op(Operand::Ref(self), Operand::Ref(rhs)).forward()
            }
        }
    };
}
