# Simple Neural Network
This repo implement a simple NN(Neural Network) for fun. The code is written in `Rust`.Development work continues...

`TODO` List
- [x] Matrix
- [x] MLP Model Inference
- [x] MLP Model BP
- [x] Read CSV
- [x] Run Mnist!
- [ ] NN Serialization
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
 ├── dataset.rs         # read mnist dataset from csv file 
 ├── layer.rs           # simple dense layer
 ├── nn.rs              # MLP based neural network 
 ├── main.rs            # MLP Mnist Demo
 └── matrix.rs          # simple implement matrix
```
## Demo in `main.rs`

```rust
use neuralnetwork::dataset::{read_csv_by_path,show_result};
use neuralnetwork::matrix::{MatrixOps};
use neuralnetwork::nn::NeuralNetwork;

fn main() {
    // read train data
    println!("Reading train data ...");
    let (train_label,train_data) = read_csv_by_path("data/mnist_train_100.csv").unwrap();

    // read test data
    println!("Reading test data ...");
    let (test_label,test_data) = read_csv_by_path("data/mnist_test_10.csv").unwrap();

    // new neural network
    let mut nn = NeuralNetwork::new(vec![784, 100, 10]);
    nn.show();

    // train
    println!("Start train ...");
    for j in 0..10 {
        println!("Epoch {}",j);
        for i in 0..train_data.len(){
            nn.train(&train_data[i].transpose(), &train_label[i].transpose());
        }
        // start eval
        println!("Start eval ...");
        for i in 0..test_data.len(){
            nn.eval(&test_data[i].transpose(), &test_label[i]);
        }
        println!("End eval");
    }
    println!("End train");

    // start eval
    println!("Start eval ...");
    for i in 0..test_data.len(){
        nn.eval(&test_data[i].transpose(), &test_label[i]);
    }
    println!("End eval");

}

```
