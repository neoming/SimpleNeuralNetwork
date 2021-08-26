use crate::matrix::{Matrix, MatrixOps};
use crate::layer::{Layer};

#[derive(Debug)]
struct NeuralNetwork {
    lr:f64,
    layers:Vec<Layer>,
}

impl NeuralNetwork {

    fn new(shape:Vec<usize>) -> NeuralNetwork{
        let mut layers = Vec::new();
        let len = shape.len();
        for i in 1..len {
            layers.push(Layer::new_by_rand(shape[i-1],shape[i]))
        }
        NeuralNetwork{
            lr:0.01,
            layers,
        }
    }
    fn inference(&self,input :Matrix) -> Matrix {
        let mut res = input;
        for layer in self.layers.iter() {
            layer.show();
            res = layer.call(&res);
            res.show();
        }
        res
    }
}

#[cfg(test)]
mod nn_tests {
    use crate::nn::NeuralNetwork;
    use crate::matrix::{Matrix,MatrixOps};

    #[test]
    fn test_inference(){
        let nn = NeuralNetwork::new(vec![3,4,5]);

        let inputs = Matrix::new(vec![vec![0.9, 0.1, 0.8]]);
        let inputs = inputs.transpose();
        nn.inference(inputs);
    }
}