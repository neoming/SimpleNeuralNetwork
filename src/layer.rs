use crate::matrix::{Matrix,MatrixOps};

#[derive(Debug)]
pub struct Layer {
    input_size:usize,
    output_size:usize,
    weights_matrix:Matrix,
}

impl Layer {
    pub fn new(data: Matrix) -> Layer{
        Layer {
            input_size:data.rows,
            output_size:data.cols,
            weights_matrix:data,
        }
    }

    pub fn new_by_rand(input_size:usize,output_size:usize)->Layer {
        Layer{
            input_size,
            output_size,
            weights_matrix: Matrix::new_by_rand(output_size,input_size)
        }
    }

    pub fn show(&self) {
        println!("[INFO] Layer input size: {}",self.input_size);
        println!("[INFO] Layer output size: {}",self.output_size);
        println!("[INFO] Layer weights matrix: ");
        self.weights_matrix.show();
    }

    pub fn call(&self,input:&Matrix) -> Matrix {
        let mut res = self.weights_matrix.product(input);
        res.activate_sigmoid();
        res
    }
}

#[cfg(test)]
mod nn_tests {
    use crate::layer::Layer;
    use crate::matrix::{Matrix,MatrixOps};

    #[test]
    fn test_show(){
        let layer = Layer::new_by_rand(3,3);
        layer.show();
    }

    #[test]
    fn test_call(){
        let weights = Matrix::new(vec![
            vec![0.9, 0.3, 0.4],
            vec![0.2, 0.8, 0.2],
            vec![0.1, 0.5, 0.6],
        ]);
        let layer = Layer::new(weights);

        let inputs = Matrix::new(vec![vec![0.9, 0.1, 0.8]]);
        let inputs = inputs.transpose();
        println!("Inputs:");
        let result = layer.call(&inputs);
        result.show();
    }
}