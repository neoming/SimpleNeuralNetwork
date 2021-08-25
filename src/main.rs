fn main() {
    let mut weights: Vec<Vec<f32>> = Vec::new();

    let mut vec1: Vec<f32> = Vec::new();
    vec1.push(0.9);
    vec1.push(0.3);
    vec1.push(0.4);
    weights.push(vec1);
    let mut vec2: Vec<f32> = Vec::new();
    vec2.push(0.2);
    vec2.push(0.8);
    vec2.push(0.2);
    weights.push(vec2);
    let mut vec3: Vec<f32> = Vec::new();
    vec3.push(0.1);
    vec3.push(0.5);
    vec3.push(0.6);
    weights.push(vec3);

    println!("Weights:");
    for line in &weights {
        for data in line {
            print!("{},", data);
        }
        print!("\n");
    }

    let mut input: Vec<f32> = Vec::new();
    input.push(0.9);
    input.push(0.1);
    input.push(0.8);

    println!("Inputs:");
    for data in &input {
        print!("{},", data);
    }
    print!("\n");

    let mut result: Vec<f32> = Vec::new();
    for line in &weights {
        let res: f32 = line.iter().zip(input.iter()).map(|(x, y)| x * y).sum();
        result.push(res);
    }

    println!("Results:");
    for data in result {
        print!("{},", data);
    }
    print!("\n");
}
