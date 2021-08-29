use neuralnetwork::dataset::read_csv_by_path;
use neuralnetwork::matrix::{Matrix,MatrixOps};
use neuralnetwork::layer::Layer;
use neuralnetwork::nn::NeuralNetwork;

fn main() {
    // read data
    let (label,data) = read_csv_by_path("data/mnist_test_10.csv").unwrap();

    // new neural network
    let mut nn = NeuralNetwork::new(vec![784, 256, 10]);
    nn.show();


    for i in 0..label.len(){
        nn.train(&data[i].transpose(), &label[i].transpose());
    }
}
