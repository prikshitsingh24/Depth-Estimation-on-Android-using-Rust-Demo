package com.example.rusttest


class RustBridge {
    external fun hello(to: String): String
    external fun square(number: Int): Int
    external fun add(number1: Int,number2: Int): Int
    external fun inference(image: ByteArray,model: ByteArray): ByteArray
}