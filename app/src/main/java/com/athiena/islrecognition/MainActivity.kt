package com.athiena.islrecognition

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.athiena.islrecognition.ui.theme.ISLRecognitionTheme
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import java.util.concurrent.Executors
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {

    private val cameraPermissionLauncher =
        registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            if (granted) {
                showCamera()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            showCamera()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun showCamera() {
        setContent {
            ISLRecognitionTheme {
                CameraScreen()
            }
        }
    }
}

@Composable
fun CameraScreen() {

    val context = LocalContext.current
    val lifecycleOwner =
        androidx.lifecycle.compose.LocalLifecycleOwner.current

    val frameBuffer = remember {
        ArrayDeque<FloatArray>()
    }

    var frameCount by remember {
        mutableIntStateOf(0)
    }

    var predictionText by remember {
        mutableStateOf("Show a greeting")
    }

    val classifier = remember {
        OnnxClassifier(context)
    }

    val handLandmarker = remember {

        HandLandmarkerHelper(
            context = context
        ) { features ->

            if (frameBuffer.size >= 91) {
                frameBuffer.removeFirst()
            }

            frameBuffer.addLast(features)

            frameCount = frameBuffer.size

            if (frameBuffer.size == 91) {

                try {

                    val prediction =
                        classifier.predict(
                            frameBuffer.toList()
                        )

                    predictionText =
                        "${prediction.label} " +
                                "${(prediction.confidence * 100).toInt()}%"

                } catch (e: Exception) {

                    e.printStackTrace()

                }
            }
        }
    }

    DisposableEffect(Unit) {

        onDispose {
            handLandmarker.close()
            classifier.close()
        }
    }

    Box(
        modifier = Modifier.fillMaxSize()
    ) {

        AndroidView(
            factory = { ctx ->

                val previewView =
                    PreviewView(ctx)

                val cameraProviderFuture =
                    ProcessCameraProvider.getInstance(ctx)

                cameraProviderFuture.addListener({

                    val cameraProvider =
                        cameraProviderFuture.get()

                    val preview =
                        Preview.Builder()
                            .build()
                            .also {
                                it.surfaceProvider =
                                    previewView.surfaceProvider
                            }

                    val imageAnalyzer =
                        ImageAnalysis.Builder()
                            .setBackpressureStrategy(
                                ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                            )
                            .build()

                    imageAnalyzer.setAnalyzer(
                        Executors.newSingleThreadExecutor()
                    ) { imageProxy ->

                        try {

                            val bitmap =
                                imageProxy.toBitmap()

                            val mpImage =
                                BitmapImageBuilder(bitmap)
                                    .build()

                            handLandmarker.detect(
                                mpImage,
                                System.currentTimeMillis()
                            )

                        } catch (e: Exception) {

                            e.printStackTrace()

                        } finally {

                            imageProxy.close()

                        }
                    }

                    val cameraSelector =
                        CameraSelector.DEFAULT_FRONT_CAMERA

                    cameraProvider.unbindAll()

                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalyzer
                    )

                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        Text(
            text = predictionText,
            fontSize = 32.sp,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 60.dp)
        )
    }
}