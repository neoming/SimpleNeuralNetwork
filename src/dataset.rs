use std::error::Error;
use crate::matrix::{Matrix,MatrixOps};

pub fn read_csv_by_path(file_path:&str) -> Result<(Vec<Matrix>,Vec<Matrix>), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(file_path)?;
    let mut label_matrix_vec = Vec::new();
    let mut data_matrix_vec = Vec::new();

    for result in rdr.records(){
        let record = result?;

        let mut label_vec = vec![0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
        let label:usize = record.get(0).unwrap().to_string().parse().unwrap();
        label_vec[label] = 1.0;
        label_matrix_vec.push(Matrix::new(vec![label_vec]));

        let mut data_vec = Vec::new();
        for i in 1..record.len(){
            let data : f64 = record.get(i).unwrap().to_string().parse().unwrap();
            data_vec.push(data/255.0 * 0.99 + 0.01)
        }
        data_matrix_vec.push(Matrix::new(vec![data_vec]))
    }
    Ok((label_matrix_vec,data_matrix_vec))
}

#[cfg(test)]
mod dataset_test {
    use super::read_csv_by_path;
    use crate::matrix::MatrixOps;

    #[test]
    fn test_read_csv_by_path(){
        println!("********[TEST] Test Dataset Read CSV By Path Function********");
        let (label,data) = read_csv_by_path("data/mnist_test_10.csv").unwrap();
        for i in 0..label.len(){
            label[i].show();
            data[i].show();
        }
        println!("********************************");
    }
}