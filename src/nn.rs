use crate::layer::Layer;
use crate::matrix::{Matrix, MatrixOps};

#[derive(Debug)]
struct NeuralNetwork {
    lr: f64,
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn new(shape: Vec<usize>) -> NeuralNetwork {
        let mut layers = Vec::new();
        let len = shape.len();
        for i in 1..len {
            layers.push(Layer::new_by_rand(shape[i - 1], shape[i]))
        }
        NeuralNetwork { lr: 0.01, layers }
    }
    fn inference(&self, input: Matrix) -> Matrix {
        let mut res = input;
        for layer in self.layers.iter() {
            layer.show();
            res = layer.call(&res);
            res.show();
        }
        res
    }

    fn train(&mut self, input: &Matrix, label: &Matrix) {
        // inference and save output
        println!("Inference");
        let mut layer_outputs = Vec::new();
        let mut res = input.clone();
        layer_outputs.push(input.clone());
        for layer in self.layers.iter() {
            res = layer.call(&res);
            // res.show();
            layer_outputs.push(res.clone());
        }
        // for output in layer_outputs.iter() { output.show(); }

        // calculate err -> update weights
        println!("Err BP");
        let mut layer_errs = Vec::new();
        let length = layer_outputs.len();
        let mut err= Matrix::zeros(1,1);
        for i in 0..length {
            let index = length - i - 1;
            if i == 0 { err = label.sub(&layer_outputs[index]); }
            else{ err = self.layers[index].weights_matrix.transpose().product(&err); }
            layer_errs.push(err.clone());
        }
        // for err in layer_errs.iter() { err.show(); }

        // update weight
        let length = layer_errs.len();
        for i in 0..length - 1 {
            let index = length - i - 1;
            let mut gradient = layer_errs[i].mul(&layer_outputs[index]);
            let mut tmp = Matrix::ones(layer_outputs[index].rows, layer_outputs[index].cols);
            tmp = tmp.sub(&layer_outputs[index]);
            gradient = tmp.mul(&gradient);
            gradient = gradient.product(&layer_outputs[index - 1].transpose());
            gradient = gradient.mul_const(self.lr);
            self.layers[index - 1].weights_matrix = self.layers[index - 1].weights_matrix.add(&gradient);
        }
    }

    fn show(&self) {
        println!("[Neural Network] learning rate: {}", self.lr);
        println!("[Neural Network] layers: ");
        for layer in self.layers.iter() {
            layer.show();
        }
    }
}

#[cfg(test)]
mod nn_tests {
    use crate::matrix::{Matrix, MatrixOps};
    use crate::nn::NeuralNetwork;

    #[test]
    fn test_inference() {
        let nn = NeuralNetwork::new(vec![3, 4, 1]);
        let inputs = Matrix::new(vec![vec![0.9, 0.1, 0.8]]);
        let inputs = inputs.transpose();
        nn.inference(inputs);
    }

    #[test]
    fn test_train() {
        let mut nn = NeuralNetwork::new(vec![3, 4, 1]);
        let inputs = Matrix::new(vec![vec![0.9, 0.1, 0.8]]);
        let label = Matrix::new(vec![vec![1.0]]);
        let inputs = inputs.transpose();
        for i in 0..10 {
            nn.show();
            nn.train(&inputs, &label);
            nn.show();
        }
    }
}
