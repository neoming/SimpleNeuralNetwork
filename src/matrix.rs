#[derive(Debug)]
pub struct Matrix {
    pub(crate) data: Vec<Vec<f64>>,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

use rand::prelude::*;
impl Clone for Matrix {
    fn clone(&self) -> Matrix {
        Matrix {
            data: self.data.clone(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

pub trait MatrixOps {
    fn new(data: Vec<Vec<f64>>) -> Matrix;
    fn new_by_rand(row: usize, col: usize) -> Matrix;
    fn zeros(row: usize, col: usize) -> Matrix;
    fn ones(row: usize, col: usize) -> Matrix;
    fn activate_sigmoid(&mut self);
    fn sigmoid(x: f64) -> f64;
    fn transpose(&self) -> Matrix;
    fn dot(&self, b: &Matrix) -> f64;
    fn dot_const(&self, b: &f64) -> f64;
    fn product(&self, b: &Matrix) -> Matrix;
    fn mul(&self, b: &Matrix) -> Matrix;
    fn mul_const(&self, b: f64) -> Matrix;
    fn add(&self, b: &Matrix) -> Matrix;
    fn sub(&self, b: &Matrix) -> Matrix;
    fn div_by_const(&self, b: f64) -> Matrix;
    fn show(&self);
}

impl MatrixOps for Matrix {
    fn new(data: Vec<Vec<f64>>) -> Matrix {
        let rows = data.len();
        assert!(rows > 0);
        let cols = data[0].len();
        assert!(cols > 0);
        Matrix { data, rows, cols }
    }

    fn new_by_rand(rows: usize, cols: usize) -> Matrix {
        assert!(rows > 0);
        assert!(cols > 0);
        let mut rand = rand::thread_rng();
        let mut data = Vec::new();
        for _row in 0..rows {
            let mut row_data = Vec::new();
            for _col in 0..cols {
                let data: f64 = rand.gen();
                row_data.push(data - 0.5);
            }
            data.push(row_data);
        }
        Matrix::new(data)
    }

    fn zeros(rows: usize, cols: usize) -> Matrix {
        assert!(rows > 0);
        assert!(cols > 0);
        let mut data = Vec::new();
        for _row in 0..rows {
            let mut row_data = Vec::new();
            for _col in 0..cols {
                row_data.push(0.0);
            }
            data.push(row_data);
        }
        Matrix::new(data)
    }

    fn ones(rows: usize, cols: usize) -> Matrix {
        assert!(rows > 0);
        assert!(cols > 0);
        let mut data = Vec::new();
        for _row in 0..rows {
            let mut row_data = Vec::new();
            for _col in 0..cols {
                row_data.push(1.0);
            }
            data.push(row_data);
        }
        Matrix::new(data)
    }

    fn activate_sigmoid(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                self.data[row][col] = Matrix::sigmoid(self.data[row][col]);
            }
        }
    }

    fn sigmoid(x: f64) -> f64 {
        let e = std::f64::consts::E;
        let res = 1.0 / (1.0 + e.powf(-1.0 * x));
        res
    }

    fn transpose(&self) -> Matrix {
        let new_row = self.cols;
        let new_col = self.rows;
        let mut new_data = Vec::new();

        for row in 0..new_row {
            let mut line = Vec::new();
            for col in 0..new_col {
                line.push(self.data[col][row]);
            }
            new_data.push(line);
        }

        Matrix {
            data: new_data,
            rows: new_row,
            cols: new_col,
        }
    }

    fn dot(&self, b: &Matrix) -> f64 {
        assert_eq!(self.rows, b.rows);
        assert_eq!(self.cols, b.cols);
        let mut res = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                res += self.data[i][j] * b.data[i][j];
            }
        }
        res
    }

    fn dot_const(&self, b: &f64) -> f64 {
        let mut res = 0.0;
        for i in 0..self.rows {
            for j in 0..self.cols {
                res += self.data[i][j] * b;
            }
        }
        res
    }

    fn product(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.cols, b.rows);
        let output_rows = self.rows;
        let output_cols = b.cols;
        let mut new_data: Vec<Vec<f64>> = Vec::new();
        for row in 0..output_rows {
            let mut new_line: Vec<f64> = Vec::new();
            for col in 0..output_cols {
                let mut res: f64 = 0.0;
                for row_self in 0..self.cols {
                    res += self.data[row][row_self] * b.data[row_self][col]
                }
                new_line.push(res);
            }
            new_data.push(new_line);
        }

        Matrix {
            data: new_data,
            rows: output_rows,
            cols: output_cols,
        }
    }

    fn mul(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.rows, b.rows);
        assert_eq!(self.cols, b.cols);
        let mut data = Vec::new();

        for row in 0..self.rows {
            let mut line = Vec::new();
            for col in 0..self.cols {
                line.push(self.data[row][col] * b.data[row][col]);
            }
            data.push(line);
        }
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn mul_const(&self, b: f64) -> Matrix {
        let mut data = Vec::new();

        for row in 0..self.rows {
            let mut line = Vec::new();
            for col in 0..self.cols {
                line.push(self.data[row][col] * b);
            }
            data.push(line);
        }
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn add(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.rows, b.rows);
        assert_eq!(self.cols, b.cols);

        let mut data = Vec::new();
        for row in 0..self.rows {
            let mut line = Vec::new();
            for col in 0..self.cols {
                line.push(self.data[row][col] + b.data[row][col]);
            }
            data.push(line);
        }
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn sub(&self, b: &Matrix) -> Matrix {
        assert_eq!(self.rows, b.rows);
        assert_eq!(self.cols, b.cols);

        let mut data = Vec::new();
        for row in 0..self.rows {
            let mut line = Vec::new();
            for col in 0..self.cols {
                line.push(self.data[row][col] - b.data[row][col]);
            }
            data.push(line);
        }
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn div_by_const(&self, b: f64) -> Matrix {
        let mut data = Vec::new();

        for row in 0..self.rows {
            let mut line = Vec::new();
            for col in 0..self.cols {
                line.push(self.data[row][col] / b);
            }
            data.push(line);
        }
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn show(&self) {
        print!(
            "[Matrix] Matrix Shape: {}x{} Data:\n[",
            self.rows, self.cols
        );
        for row in 0..self.rows {
            print!("[");
            for col in 0..self.cols {
                print!("{}", self.data[row][col]);
                if col != self.cols - 1 {
                    print!(",");
                }
            }
            print!("]");
            if row != self.rows - 1 {
                print!(",\n");
            }
        }
        print!("]\n");
    }
}

#[cfg(test)]
mod matrix_tests {

    use super::Matrix;
    use crate::matrix::MatrixOps;

    #[test]
    fn test_show() {
        println!("********[TEST] Test Matrix Show Function********");
        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let row1: Vec<f64> = vec![0.2, 0.8, 0.2];
        let row2: Vec<f64> = vec![0.1, 0.5, 0.6];
        let data: Vec<Vec<f64>> = vec![row0, row1, row2];

        let matrix: Matrix = Matrix::new(data);
        matrix.show();
        println!("********************************");
    }

    #[test]
    fn test_product() {
        println!("********[TEST] Test Matrix Product Function********");
        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let row1: Vec<f64> = vec![0.2, 0.8, 0.2];
        let row2: Vec<f64> = vec![0.1, 0.5, 0.6];
        let data: Vec<Vec<f64>> = vec![row0, row1, row2];
        let matrix0: Matrix = Matrix::new(data);
        matrix0.show();

        let row0: Vec<f64> = vec![0.9];
        let row1: Vec<f64> = vec![0.1];
        let row2: Vec<f64> = vec![0.8];
        let data: Vec<Vec<f64>> = vec![row0, row1, row2];
        let matrix1: Matrix = Matrix::new(data);
        matrix1.show();

        let matrix2: Matrix = matrix0.product(&matrix1);
        matrix2.show();
        println!("********************************");
    }

    #[test]
    fn test_transpose() {
        println!("********[TEST] Test Matrix Transpose Function********");
        let row0: Vec<f64> = vec![0.9];
        let row1: Vec<f64> = vec![0.1];
        let row2: Vec<f64> = vec![0.8];
        let data: Vec<Vec<f64>> = vec![row0, row1, row2];

        let matrix1: Matrix = Matrix::new(data);
        matrix1.show();

        let matrix2: Matrix = matrix1.transpose();
        matrix2.show();
        assert_eq!(matrix1.cols, matrix2.rows);
        assert_eq!(matrix1.rows, matrix2.cols);
        println!("********************************");
    }

    #[test]
    fn test_dot() {
        println!("********[TEST] Test Matrix Dot Function********");
        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let data: Vec<Vec<f64>> = vec![row0];
        let matrix0: Matrix = Matrix::new(data);
        matrix0.show();

        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let data: Vec<Vec<f64>> = vec![row0];
        let matrix1: Matrix = Matrix::new(data);
        matrix1.show();

        let res = matrix0.dot(&matrix1);
        println!("{}", res);
        println!("********************************");
        assert_eq!(res, 1.06);
    }

    #[test]
    fn test_dot_const() {
        println!("********[TEST] Test Matrix Dot Const Function********");
        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let data: Vec<Vec<f64>> = vec![row0];
        let matrix0: Matrix = Matrix::new(data);
        matrix0.show();

        let val = 0.5;
        let res = matrix0.dot_const(&val);
        println!("{}", res);
        println!("********************************");
        assert_eq!(res, 0.8);
    }

    #[test]
    fn test_mul() {
        println!("********[TEST] Test Matrix Mul Function********");
        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let data: Vec<Vec<f64>> = vec![row0];
        let matrix0: Matrix = Matrix::new(data);
        matrix0.show();

        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let data: Vec<Vec<f64>> = vec![row0];
        let matrix1: Matrix = Matrix::new(data);
        matrix1.show();

        let matrix2: Matrix = matrix0.mul(&matrix1);
        matrix2.show();
        println!("********************************");
    }

    #[test]
    fn test_mul_const() {
        println!("********[TEST] Test Matrix Mul Const Function********");
        let row0: Vec<f64> = vec![0.9, 0.3, 0.4];
        let data: Vec<Vec<f64>> = vec![row0];
        let matrix0: Matrix = Matrix::new(data);
        matrix0.show();

        let b = 0.5;

        let matrix1: Matrix = matrix0.mul_const(b);
        matrix1.show();
        println!("********************************");
    }

    #[test]
    fn test_activate_sigmoid() {
        println!("********[TEST] Test Matrix Activate Sigmoid Function********");
        let row0: Vec<f64> = vec![0.975, 0.888, 1.254];
        let data: Vec<Vec<f64>> = vec![row0];
        let mut matrix0: Matrix = Matrix::new(data);
        matrix0.show();
        matrix0.activate_sigmoid();
        matrix0.show();
        println!("********************************");
    }

    #[test]
    fn test_new_by_rand() {
        println!("********[TEST] Test Matrix New By Rand Function********");
        let a: Matrix = Matrix::new_by_rand(3, 3);
        a.show();
        println!("********************************");
    }

    #[test]
    fn test_add() {
        println!("********[TEST] Test Matrix Add Function********");
        let a: Matrix = Matrix::new_by_rand(3, 3);
        a.show();
        let b: Matrix = Matrix::new_by_rand(3, 3);
        b.show();
        let c = a.add(&b);
        c.show();
        println!("********************************");
    }

    #[test]
    fn test_sub() {
        println!("********[TEST] Test Matrix Sub Function********");
        let a: Matrix = Matrix::new_by_rand(3, 3);
        a.show();
        let b: Matrix = Matrix::new_by_rand(3, 3);
        b.show();
        let c = a.sub(&b);
        c.show();
        println!("********************************");
    }

    #[test]
    fn test_div_const() {
        println!("********[TEST] Test Matrix Div By Const Function********");
        let a: Matrix = Matrix::new_by_rand(3, 3);
        a.show();
        let c = a.div_by_const(2.0);
        c.show();
        println!("********************************");
    }
}
