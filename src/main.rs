use std::string;
use rand::prelude::*;
use image::*;
use crate::data1::*;
use crate::data2::*;
mod data1;
mod data2;
static THRESHOLD:f32 = 0.0;

// Draws the image corresponding to the 7 column 9 row matrix
// represented as a 2D-array and saves it to output_images  
// folder as a .png file with the given filename.
fn draw(matrix:[[i32;7];9],path:&str){
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
                panic!("Could not parse pixel color at ({},{}) in {}",x,y,path);
            }
        }
    }
    // Save the image to the output_images folder
    image_buffer.save(&path).unwrap();

}       

//Bipolar activation function based on the const threshold value
fn activation(input:f32)->i32 {
    if input >= THRESHOLD {
        return 1;
    }
    else{
        return -1;
    }
}


fn train_heteroassociative(weights:& mut [[[[f32;7];9];7];9],s:&[[i32;7];9],t:&[[i32;7];9]){
    // For each pixel in the input
    for x in 0..9{
        for y in 0..7{
            // For each pixel in the output
            for j in 0..9{
                for k in 0..7 {
                    // Updating the (j,k)th weight corresponding to the (x,y)th input neuron
                    weights[x][y][j][k] = weights[x][y][j][k] + (s[x][y] * t[j][k]) as f32;
                }
            }  
        }
    }
}

fn train_autoassociative(weights:& mut [[[[f32;7];9];7];9],s:&[[i32;7];9]){
    for x in 0..9{
        for y in 0..7{
            // For each pixel in the output
            for j in 0..9{
                for k in 0..7 {
                    // Updating the (j,k)th weight corresponding to the (x,y)th input neuron
                    weights[x][y][j][k] = weights[x][y][j][k] + (s[x][y] * s[j][k]) as f32;
                }
            }  
        }
    }
}


// Calculates the output matrix of the network given it's current state and an input
fn get_output(weights:& mut [[[[f32;7];9];7];9],s:&[[i32;7];9]) ->[[i32;7];9]{
    let mut output:[[i32;7];9] = [[0;7];9];
    // For every pixel in the output
    for x in 0..9{
        for y in 0..7{
            let mut sum:i32 = 0;
            // For every pixel in the input
            for j in 0..9{
                for k in 0..7{
                    sum = sum + s[j][k] * weights[j][k][x][y] as i32;
                }
            }
            //output[x][y] = activation((sum + biases[x][y] as i32)as f32);
            output[x][y] = activation((sum)as f32);
   
        }
    }
    return output
}
    
fn main() {
    // 3-D array storing the set of all weights in the network
    // There are 63 inputs each with its own 7x9 array of weights
    // connecting them to the output neurons.
    let mut weights:[[[[f32;7];9];7];9] = [[[[0.0;7];9];7];9];
    // Randomizing weights...
    let mut r = StdRng::seed_from_u64(42);
    for x in 0..9 {
        for y in 0..7{
            for j in 0..9{
                for k in 0..7{
                    weights[x][y][j][k] = r.gen_range(-1.0..1.0);
                }
            }
        }
    }

    // Part 1 (A)
    println!("Generating images for Part 1-A");
    // Drawing the expected inputs and outputs
    draw(S1,"./Part1_A/input_images/s1.png");
    draw(S2,"./Part1_A/input_images/s2.png");
    draw(S3, "./Part1_A/input_images/s3.png");

    //Training the network on the expected input/output pairs
    println!("Training heteroassociative network...");
    train_heteroassociative(&mut weights,&S1,&T1);
    train_heteroassociative(&mut weights,&S2,&T2);
    train_heteroassociative(&mut weights,&S3,&T3);

    // Putting the inputs into the network and drawing what comes out
    let S1_out:[[i32;7];9] = get_output(&mut weights,&S1);
    let S2_out:[[i32;7];9] = get_output(&mut weights,&S2);
    let S3_out:[[i32;7];9] = get_output(&mut weights,&S3);
    draw(S1_out,"./Part1_A/output_images/S1.png");
    draw(S2_out,"./Part1_A/output_images/S2.png");
    draw(S3_out,"./Part1_A/output_images/S3.png");
    println!("Re-training heteroassociative network...");
    println!("Generating images for Part 1-B");
    //Part 1 (B)
    // Re-setting the weights back to zero
    let mut weights:[[[[f32;7];9];7];9] = [[[[0.0;7];9];7];9];
    // Re-training network
    
    train_heteroassociative(&mut weights,&S1,&T1);
    train_heteroassociative(&mut weights,&S2,&T2);
    train_heteroassociative(&mut weights,&S3,&T3);
    train_heteroassociative(&mut weights,&S4_custom,&T4_custom);
    train_heteroassociative(&mut weights,&S5_custom,&T5_custom);
    // Getting output images
    let S1_out:[[i32;7];9] = get_output(&mut weights,&S1);
    let S2_out:[[i32;7];9] = get_output(&mut weights,&S2);
    let S3_out:[[i32;7];9] = get_output(&mut weights,&S3);
    let S4_custom_out:[[i32;7];9] = get_output(&mut weights,&S4_custom);
    let S5_custom_out:[[i32;7];9] = get_output(&mut weights,&S5_custom);

    //Drawing input-output pairs
    draw(S1,"./Part1_B/input_images/S1.png");
    draw(S2,"./Part1_B/input_images/S2.png");
    draw(S3,"./Part1_B/input_images/S3.png");
    draw(S4_custom,"./Part1_B/input_images/S4_custom.png");
    draw(S5_custom,"./Part1_B/input_images/S5_custom.png");

    draw(S1_out,"./Part1_B/output_images/S1.png");
    draw(S2_out,"./Part1_B/output_images/S2.png");
    draw(S3_out,"./Part1_B/output_images/S3.png");
    draw(S4_custom_out,"./Part1_B/output_images/S4_custom.png");
    draw(S5_custom_out,"./Part1_B/output_images/S5_custom.png");

    // Part1 (C)
    println!("Generating images with noise for Part 1-C");
    draw(S1_noisy,"./Part1_C/input_images/S1_noisy.png");
    draw(S4_custom_noisy,"./Part1_C/input_images/S4_custom_noisy.png");
    draw(S5_custom_noisy,"./Part1_C/input_images/S5_custom_noisy.png");

    // Getting output images
    // Getting output images
    let S1_noisy_out:[[i32;7];9] = get_output(&mut weights,&S1_noisy);
    let S4_custom_noisy_out:[[i32;7];9] = get_output(&mut weights,&S4_custom_noisy);
    let S5_custom_noisy_out:[[i32;7];9] = get_output(&mut weights,&S5_custom_noisy);

    //Drawing outputs
    draw(S1_noisy_out,"./Part1_C/output_images/S1_noisy.png");
    draw(S4_custom_noisy_out,"./Part1_C/output_images/S4_custom_noisy.png");
    draw(S5_custom_noisy_out,"./Part1_C/output_images/S5_custom_noisy.png");
    println!("Training autoassociative network...");
    println!("Generating images for Part 2- A & B");
    // Part2 (A & B)
    let mut weights:[[[[f32;7];9];7];9] = [[[[0.0;7];9];7];9];
    // Training a network on the second dataset using the autoassociative hebb rule
    train_autoassociative(&mut weights, &S1_2);
    train_autoassociative(&mut weights, &S2_2);
    train_autoassociative(&mut weights, &S3_2);
    train_autoassociative(&mut weights, &S4_2);
    train_autoassociative(&mut weights, &S5_2);
    train_autoassociative(&mut weights, &S6_2);
    train_autoassociative(&mut weights, &S7_2);
    train_autoassociative(&mut weights, &S8_2);
    train_autoassociative(&mut weights, &S9_2);
    train_autoassociative(&mut weights, &S10_2);
    // Calculating the outputs for all 10 images
    let S1_out:[[i32;7];9] = get_output(&mut weights,&S1_2);
    let S2_out:[[i32;7];9] = get_output(&mut weights,&S2_2);
    let S3_out:[[i32;7];9] = get_output(&mut weights,&S3_2);
    let S4_out:[[i32;7];9] = get_output(&mut weights,&S4_2);
    let S5_out:[[i32;7];9] = get_output(&mut weights,&S5_2);
    let S6_out:[[i32;7];9] = get_output(&mut weights,&S6_2);
    let S7_out:[[i32;7];9] = get_output(&mut weights,&S7_2);
    let S8_out:[[i32;7];9] = get_output(&mut weights,&S8_2);
    let S9_out:[[i32;7];9] = get_output(&mut weights,&S9_2);
    let S10_out:[[i32;7];9] = get_output(&mut weights,&S10_2);
    // Drawing the input-output pairs for all 10 images
 
    draw(S1_2,"./Part2/S1.png");
    draw(S1_out,"./Part2/S1_out.png");
    draw(S2_2,"./Part2/S2.png");
    draw(S2_out,"./Part2/S2_out.png");
    draw(S3_2,"./Part2/S3.png");
    draw(S3_out,"./Part2/S3_out.png");
    draw(S4_2,"./Part2/S4.png");
    draw(S4_out,"./Part2/S4_out.png");
    draw(S5_2,"./Part2/S5.png");
    draw(S5_out,"./Part2/S5_out.png");
    draw(S6_2,"./Part2/S6.png");
    draw(S6_out,"./Part2/S6_out.png");
    draw(S7_2,"./Part2/S7.png");
    draw(S7_out,"./Part2/S7_out.png");
    draw(S8_2,"./Part2/S8.png");
    draw(S8_out,"./Part2/S8_out.png");
    draw(S9_2,"./Part2/S9.png");
    draw(S9_out,"./Part2/S9_out.png");
    draw(S10_2,"./Part2/S10.png");
    draw(S10_out,"./Part2/S10_out.png");
    
    //Part 2 (C)
    println!("Generating images for Part 2-C");
    for i in 0..5{
        let S1_out:[[i32;7];9] = get_output(&mut weights,&S1_out);
        let path = format!("./Part2/S1_out_iter{}.png",i);
        draw(S1_out,&path);
    }
    println!("Done!");


}
