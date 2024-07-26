#[macro_export]
macro_rules! backward {
    ($v1:ident) => {
        if let Operand::Ref(ref v) = $v1 {
            if let Some(g) = v.grad() {
                v._backward(g);
            }
        }
    };
    ($v1:ident, $v2:ident) => {
        match ($v1, $v2) {
            (Operand::Ref(v1), Operand::Ref(v2)) => {
                if let Some(g) = v1.grad() {
                    v1._backward(g);
                };
                if let Some(g) = v2.grad() {
                    if !std::ptr::eq(*v1, *v2) {
                        v2._backward(g);
                    }
                }
            }
            (Operand::Ref(v), Operand::Const(_)) => {
                if let Some(g) = v.grad() {
                    v._backward(g);
                };
            }
            (Operand::Const(_), Operand::Ref(v)) => {
                if let Some(g) = v.grad() {
                    v._backward(g);
                };
            }
            (Operand::Const(_), Operand::Const(_)) => {}
        }
    };
}
