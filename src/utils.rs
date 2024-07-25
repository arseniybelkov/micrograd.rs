#[macro_export]
macro_rules! if_req_grad {
    ($ident:ident, $expr:expr) => {
        if $ident.requires_grad {
            $expr;
        }
    };
}

#[macro_export]
macro_rules! unwrap_operands {
    ($v1:ident, $update1:stmt, $v2:ident, $update2:stmt) => {
        match ($v1, $v2) {
            (Operand::Ref(v1), Operand::Ref(v2)) => {
                $update1
                v1._backward(v1.grad());
                $update2
                if !std::ptr::eq(v1, v2) {
                    v2._backward(v2.grad());
                }
            }
            (Operand::Ref(v1), Operand::Const(v2)) => {
                $update1
                v1._backward(v1.grad());
            }
            (Operand::Const(v1), Operand::Ref(v2)) => {
                $update2
                v2._backward(v2.grad());
            }
            (Operand::Const(v1), Operand::Const(v2)) => return,
        }
    };
}
