
#![cfg(target_os="android")]
#![allow(non_snake_case)]

use jni::sys::jbyteArray;
use ndarray::Array3;
use ndarray::Array4;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::slice;
use std::ffi::{CString, CStr};
use jni::JNIEnv;
use jni::objects::{JObject, JString};
use jni::sys::{jstring,jint};
use std::panic::catch_unwind;
use jni::objects::JClass;
use std::fs;
use jni::sys::{jbyte, jsize, JNI_ABORT};
use ort::value::Value;
use image::{GrayImage, Luma};
use std::io::Cursor;
use std::error::Error;

#[unsafe(no_mangle)]
pub extern "C" fn Java_com_example_rusttest_RustBridge_hello(
    env: JNIEnv,
    _: JObject,
    j_recipient: JString
) -> jstring {
    // Convert Java string to Rust string
    let recipient = match env.get_string(j_recipient) {
        Ok(string) => string.to_string_lossy().into_owned(),
        Err(_) => "Error getting string".to_string()
    };

    // Create the greeting message
    let greeting = format!("Hello {}!", recipient);

    // Convert Rust string back to Java string
    let output = env.new_string(greeting)
        .expect("Failed to create Java string");

    // Return as raw JNI string pointer
    output.into_inner()
}

#[unsafe(no_mangle)]
pub extern "C" fn Java_com_example_rusttest_RustBridge_square(
    env: jni::JNIEnv,
    _: jni::objects::JObject,
    number: jint,  // This is the type for `int` in Java/Kotlin
) -> jint {
    // Squaring the number
    let result = number * number;

    // Return the result back to Kotlin/Java
    result
}

#[unsafe(no_mangle)]
pub extern "C" fn Java_com_example_rusttest_RustBridge_add(
    env: jni::JNIEnv,
    _: jni::objects::JObject,
    number1: jint,  // This is the type for `int` in Java/Kotlin
    number2: jint
) -> jint {

    let result = number1 + number2;

    // Return the result back to Kotlin/Java
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Java_com_example_rusttest_RustBridge_inference(
    env: jni::JNIEnv,
    _class: jni::objects::JClass,
    image: jni::sys::jbyteArray,
    model: jni::sys::jbyteArray,
) -> jni::sys::jbyteArray {
    // Use catch_unwind to prevent Rust panics from crashing the entire app
    let result = catch_unwind(|| {
        // Get the byte array from JNI for the image
        let image_data = match env.convert_byte_array(image) {
            Ok(data) => data,
            Err(_) => {
                eprintln!("Failed to retrieve image byte array from JNI");
                return env.byte_array_from_slice(&[0u8, 0u8, 0u8]) // Return error as byte array (e.g., [0, 0, 0] as an error code)
                    .expect("Failed to create byte array");
            }
        };

        log_error(&env, &format!("Inference started"));

        // Process with ONNX model
        match process_image(&image_data, model, &env) {
            Ok(prediction) => {
                println!("Image processed successfully");
                // Return prediction as jbyteArray
                env.byte_array_from_slice(&prediction) 
                    .expect("Failed to create byte array")
            }
            Err(e) => {
                eprintln!("Error processing image: {:?}", e);
                env.byte_array_from_slice(&[0u8, 2u8]) 
                    .expect("Failed to create byte array")
            }
        }
    });

    match result {
        Ok(status) => {
            status // Return the result as a jbyteArray
        }
        Err(error) => {
            log_error(&env, &format!("A panic occurred during inference: {:?}", error));
            env.byte_array_from_slice(&[0u8, 0u8, 1u8])  // Return error code byte array in case of panic
                .expect("Failed to create byte array")
        }
    }
}



fn log_error(env: &JNIEnv, message: &str) {
    let java_class = "android/util/Log";
    let log_method = "e";
    let tag = "RustError"; // Use a specific tag for Rust errors

    // Convert strings to JString and then to JValue
    let tag_string = env.new_string(tag).unwrap();
    let message_string = env.new_string(message).unwrap();

    // Call Log.e in Android to log the error message
    env.call_static_method(
        java_class,
        log_method,
        "(Ljava/lang/String;Ljava/lang/String;)I",
        &[
            (*tag_string).into(),  // Convert JString to JValue
            (*message_string).into(), // Convert JString to JValue
        ],
    )
    .unwrap();
}



fn process_image(image: &Vec<u8>, model: jni::sys::jbyteArray, env: &JNIEnv) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    println!("Initializing model...");

    // Dynamically calculate the image dimensions
    let total_pixels = image.len() / 3; // Assuming the image is in RGB format (3 channels)
    
    let (original_width, original_height) = find_image_dimensions(total_pixels)?;

    log_error(&env,&format!("Image dimensions: {}x{}", original_width, original_height));
    log_error(&env,&format!("Actual bytes received: {}", image.len()));

    // Target dimensions for model input
    let target_width = 518;
    let target_height = 518;

    // Create padded image (if resizing is required)
    let mut padded_image = vec![0u8; target_width * target_height * 3];

    // Calculate starting positions to center the original image
    let start_x = (target_width - original_width) / 2;
    let start_y = (target_height - original_height) / 2;

    // Copy original image data to center of padded image
    for y in 0..original_height {
        for x in 0..original_width {
            for c in 0..3 { // 3 channels (RGB)
                let orig_idx = (y * original_width + x) * 3 + c;
                let pad_idx = ((start_y + y) * target_width + (start_x + x)) * 3 + c;
                padded_image[pad_idx] = image[orig_idx];
            }
        }
    }

    // Normalize the image to [0, 1] by converting to Float32
    let mut normalized_image = vec![0.0f32; target_width * target_height * 3];
    for (i, &pixel) in padded_image.iter().enumerate() {
        normalized_image[i] = pixel as f32 / 255.0; // Normalize to [0, 1]
    }

    // Convert the normalized image to 4D tensor (batch_size, channels, height, width)
    let input_tensor = Array4::from_shape_vec(
        (1, 3, target_height, target_width),
        normalized_image.clone(),
    ).map_err(|e| {
        format!(
            "Failed to reshape padded image. Total length: {}, Shape: (1, 3, {}, {}). Error: {:?}",
            padded_image.len(), target_height, target_width, e
        )
    })?;

    // Convert ndarray to ONNX Runtime Value (Float32)
    let input_value = Value::from_array(input_tensor)?;

    // Log the input value for debugging purposes
    log_error(&env, &format!("The input value to the model ----{:?}", input_value));

    // Load the model
    let model_data = load_model(model, env);
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_memory(&model_data)?;

    log_error(&env, &format!("Base model ----{:?}", model));

    // Run model inference
    let outputs = model.run(ort::inputs!{"pixel_values" => input_value}?)?;
    log_error(&env, &format!("The model output ----{:?}", outputs));
    let prediction = outputs["predicted_depth"].try_extract_tensor::<f32>()?;
    
    // Flatten the prediction tensor to Vec<u8> (if needed for further processing)
    let prediction_flatten: Vec<u8> = prediction.iter().map(|&x| (x * 255.0) as u8).collect(); // Rescale to [0, 255] range
    log_error(&env, &format!("Flattened prediction ----{:?}", prediction_flatten));
    println!("Model prediction successful");

    let min_val = prediction.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = prediction.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut img = GrayImage::new(target_width as u32, target_height as u32);
    
    for (i, &value) in prediction.iter().enumerate() {
        let normalized = ((value - min_val) / (max_val - min_val) * 255.0) as u8;
        let x = (i % target_width) as u32;
        let y = (i / target_width) as u32;
        img.put_pixel(x, y, Luma([normalized]));
    }

    // Encode the image as PNG
    let mut buffer = Vec::new();
    img.write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png)?;

    println!("Image conversion successful");

    Ok(buffer)  // Return PNG bytes

}


fn find_image_dimensions(total_pixels: usize) -> Result<(usize, usize), Box<dyn std::error::Error>> {

    let possible_dimensions = vec![
        (512, 512),    // Square aspect ratio
        (640, 480),    // Common 4:3 ratio
        (800, 600),    // Common 4:3 ratio
        (1024, 768),   // Common 4:3 ratio
        (1920, 1080),  // Full HD 16:9 ratio
    ];

    for (width, height) in possible_dimensions {
        if width * height == total_pixels {
            return Ok((width, height));
        }
    }

    // If no known dimensions match, attempt to calculate based on the number of pixels
    let width = (total_pixels as f64).sqrt().round() as usize;
    let height = total_pixels / width;

    Ok((width, height))
}



fn load_model(
    model: jni::sys::jbyteArray,
    env: &JNIEnv, 
) -> Vec<u8> {
    match env.convert_byte_array(model) {
        Ok(data) => data,
        Err(_) => {
            eprintln!("Failed to retrieve byte array from JNI");
            Vec::new() // Return an empty Vec if there's an error
        }
    }
}


