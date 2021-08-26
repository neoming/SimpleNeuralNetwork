use neuralnetwork::matrix::*;

fn main() {
    let weights = Matrix::new(vec![
        vec![0.9, 0.3, 0.4],
        vec![0.2, 0.8, 0.2],
        vec![0.1, 0.5, 0.6],
    ]);
    println!("Weights:");
    weights.show();

    let inputs = Matrix::new(vec![vec![0.9, 0.1, 0.8]]);
    let inputs = inputs.transpose();
    println!("Inputs:");
    inputs.show();

    let result = weights.product(&inputs);
    println!("Results:");
    result.show();
}
