# Simple Neural Network
This repo implement a simple NN(Neural Network) for fun. The code is written in `Rust`.Development work continues...

`TODO` List
- [x] Matrix
- [ ] MLP Model Inference
- [ ] MLP Model BP
## How to run this code
1. You need to install rust on your pc
2. Clone this repo
3. `cargo test` -> `cargo run`!

## Project Structure
```shell
├── Cargo.lock
├── Cargo.toml
├── README.md
├── src               # source code
 ├── lib.rs             # mod 
 ├── main.rs            # matrix demo
 └── matrix.rs          # simple implement matrix
```
### `Matrix`  
`Matrix`
```rust
pub struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}
```

`Matrix trait`
```rust
trait MatrixOps {
    fn new(data:Vec<Vec<64>>) -> Matrix;
    fn new_by_rand(row: usize, col: usize) -> Matrix;
    fn activate_sigmoid(&mut self);
    fn sigmoid(x:f64) -> f64;
    fn transpose(&self) -> Matrix;
    fn dot(&self, b: &Matrix) ->  f64;
    fn dot_const(&self, b: &f64) ->  f64;
    fn product(&self, b: &Matrix ) -> Matrix;
    fn mul(&self, b: &Matrix) -> Matrix;
    fn mul_const(&self,b:f64) -> Matrix;
    fn show(&self);
}
```