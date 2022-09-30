use std::string;

use image::*;
use crate::data1::*;
mod data1;

static THRESHOLD:f32 = 1.0;



// Draws the image corresponding to the 7 column 9 row matrix
// represented as a 2D-array and saves it to output_images as 
// a .png file with the given filename.
fn draw(matrix:[[i32;7];9],filename:&str){
    let mut image_buffer = image::ImageBuffer::new(9,7);
    for x in 0..9 {
        for y in 0..7 {
            let pixel = image_buffer.get_pixel_mut(x,y);

            if matrix[x as usize][y as usize] == 1 {
                // If the matrix entry is 1, color pixel black.
                *pixel = image::Rgb([0.0 as u8,0.0 as u8,0.0 as u8]);
                               
            }
            else if matrix[x as usize][y as usize] == -1{
                // If the matrix entry is -1, color pixel white.
                *pixel = image::Rgb([255.0 as u8,255.0 as u8,255.0 as u8]);
                
            }
            else{
                panic!();
            }
        }
    }

    let path = format!("./output_images/{}.png",filename);
    // Save the image to the output_images folder
    image_buffer.save(&path).unwrap();
    let img = image::open(&path).unwrap();
    img.resize(700,900,imageops::Gaussian);
    img.save(&path).unwrap();

}       

//Bipolar activation function based on the threshold value
fn activation(input:f32)->f32 {
    if input >= THRESHOLD {
        return 1.00;
    }
    else{
        return 0.00;
    }
}


fn train(weights:& mut [[[f32;7];9];63], biases: &mut [[f32;7];9],s:&[[i32;7];9],t:&[[i32;7];9]){
    for x in 0..9{
        for y in 0..7{
            for i in 0..63{
                weights[i][x][y] = weights[i][x][y] + (s[x][y] * t[x][y]) as f32;
            }
            biases[x][y] = biases[x][y] + t[x][y] as f32;
        }   
    }
}
    
fn main() {
    // 3-D array storing the set of all weights in the network
    // There are 63 inputs each with its own 7x9 array of weights
    // connecting them to the output neurons.
    let mut weights:[[[f32;7];9];63] = [[[0.0;7];9];63];
    // array of biases, there is one bias for each of the 63 output neurons
    let mut biases:[f32;63] = [[0.0;7];9];





    //let mut weights:[i32;9];
    draw(S1,stringify!(S1));
    draw(S2,stringify!(S2));
    draw(S3, stringify!(S3));

    draw(T1, stringify!(T1));
    draw(T2, stringify!(T2));
    draw(T3, stringify!(T3));

}
