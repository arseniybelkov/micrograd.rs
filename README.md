# micrograd.rs
> _Quit your shitty ass job and go learn some skills._  
> 
> _-_ George Hotz

Insanely lightweight Rust implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd). 
## Example 
```rust
use micrograd::Value;

fn main() {
    let x = Value::new(1f32);
    let y = Value::new(2f32);
    
    let z = &x * &y;
    
    z.backward();
    assert_eq!(x.grad(), 2f32);
    assert_eq!(y.grad(), 1f32);
}
```
## Supported types
`Value<T>` requires `T` to implement [Differentiable](https://github.com/arseniybelkov/micrograd.rs/blob/master/src/differentiable.rs). 
You can support your own types by implementing `micrograd::Differentiable` for them.

## Features
This crate performs no heap allocations, all the values and the graph are
stored entirely on stack. Thus graph requires all the nodes to be alive during
`backward`, otherwise the code will not compile since borrow checker does 
not allow invalid references.
