#[macro_export]
macro_rules! if_req_grad {
    ($ident:ident, $expr:expr) => (
        if $ident.requires_grad {
            $expr;
        }
    )
}
