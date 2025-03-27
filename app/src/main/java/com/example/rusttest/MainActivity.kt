package com.example.rusttest

import android.annotation.SuppressLint
import android.content.Intent
import android.content.res.AssetManager
import android.database.Cursor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.drawable.BitmapDrawable
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultCallback
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import java.io.ByteArrayOutputStream
import java.io.FileDescriptor
import java.io.IOException
import java.io.InputStream

class MainActivity : ComponentActivity() {
    private var galleryActivityResultLauncher: ActivityResultLauncher<Intent> = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult(), ActivityResultCallback {
            if (it.getResultCode() === RESULT_OK) {
                image_uri = it.data?.data
                val inputImage = uriToBitmap(image_uri!!)
                val rotated = rotateBitmap(inputImage!!)
                originalImageView.setImageBitmap(rotated)
            }
        }
    )


    lateinit var originalImageView: ImageView;
    lateinit var generatedImageView: ImageView;
    lateinit var filePickerBtn: Button;
    lateinit var generateBtn:Button;
    var image_uri: Uri? = null;
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        System.loadLibrary("rust")
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        originalImageView = findViewById(R.id.original);
        generatedImageView = findViewById(R.id.generated);
        filePickerBtn = findViewById(R.id.fliePickerbtn);
        generateBtn = findViewById(R.id.generateBtn);

        filePickerBtn.setOnClickListener {
            val galleryIntent =
                Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            galleryActivityResultLauncher.launch(galleryIntent)
        }

        generateBtn.setOnClickListener {
            val bitmap = (originalImageView.getDrawable() as BitmapDrawable).getBitmap();
            val stream = ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.PNG, 90, stream);
            val imageByteArray = stream.toByteArray();
            val assetManager = assets // Access the AssetManager to load the model
            val modelData = loadModelFromAssets(assetManager, "depth-anythingv2-base.onnx") // Load model file from assets

            if (modelData != null) {
                var rustBridge = RustBridge();
                val result = rustBridge.inference(imageByteArray,modelData);
                printByteArrayProperties(result)
                val resultBitmap = BitmapFactory.decodeByteArray(result, 0, result.size)
                generatedImageView.setImageBitmap(resultBitmap);
            } else {
                println("Model loading failed!")
            }
        }

    }

    fun printByteArrayProperties(byteArray: ByteArray) {
        println("ByteArray Properties:")
        println("Size (in bytes): ${byteArray.size}")

        // Print raw byte array (first 20 bytes for brevity)
        println("First 20 raw bytes: ${byteArray.take(20).joinToString(", ") { it.toString() }}")

        // Print byte array as Hexadecimal (first 20 bytes for brevity)
        val hexString = byteArray.take(20).joinToString(" ") { String.format("%02X", it) }
        println("First 20 bytes in Hexadecimal: $hexString")

        // Full byte array in Hexadecimal (not recommended for large arrays, but here's how it looks)
        val fullHexString = byteArray.joinToString(" ") { String.format("%02X", it) }
        println("Full Byte Array in Hexadecimal: $fullHexString")

        // Full byte array in raw form (first 20 for brevity)
        println("Full Byte Array raw form (first 20 bytes): ${byteArray.joinToString(", ")}")
    }

    private fun uriToBitmap(selectedFileUri: Uri): Bitmap? {
        try {
            val parcelFileDescriptor = contentResolver.openFileDescriptor(selectedFileUri, "r")
            val fileDescriptor: FileDescriptor = parcelFileDescriptor!!.fileDescriptor
            val image = BitmapFactory.decodeFileDescriptor(fileDescriptor)
            parcelFileDescriptor.close()
            return image
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return null
    }

    //TODO rotate image if image captured on samsung devices
    //TODO Most phone cameras are landscape, meaning if you take the photo in portrait, the resulting photos will be rotated 90 degrees.
    @SuppressLint("Range")
    fun rotateBitmap(input: Bitmap): Bitmap? {
        val orientationColumn =
            arrayOf(MediaStore.Images.Media.ORIENTATION)
        val cur: Cursor? = contentResolver.query(image_uri!!, orientationColumn, null, null, null)
        var orientation = -1
        if (cur != null && cur.moveToFirst()) {
            orientation = cur.getInt(cur.getColumnIndex(orientationColumn[0]))
        }
        Log.d("tryOrientation", orientation.toString() + "")
        val rotationMatrix = Matrix()
        rotationMatrix.setRotate(orientation.toFloat())
        return Bitmap.createBitmap(input, 0, 0, input.width, input.height, rotationMatrix, true)
    }
}

fun loadModelFromAssets(assetManager: AssetManager, modelName: String): ByteArray? {
    return try {
        val inputStream: InputStream = assetManager.open(modelName) // Load model from assets
        val byteArrayOutputStream = ByteArrayOutputStream()
        val buffer = ByteArray(1024)
        var length: Int
        while (inputStream.read(buffer).also { length = it } != -1) {
            byteArrayOutputStream.write(buffer, 0, length)
        }
        inputStream.close()
        byteArrayOutputStream.toByteArray()
    } catch (e: IOException) {
        e.printStackTrace()
        null
    }
}



