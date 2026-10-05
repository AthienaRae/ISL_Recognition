package com.athiena.islrecognition

import android.content.Context
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import kotlin.math.sqrt

class HandLandmarkerHelper(
    context: Context,
    private val onResult: (FloatArray) -> Unit
) {

    private val handLandmarker: HandLandmarker

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(2)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { result, _ ->
                processResult(result)
            }
            .build()

        handLandmarker =
            HandLandmarker.createFromOptions(context, options)
    }

    fun detect(
        image: MPImage,
        timestampMs: Long
    ) {
        handLandmarker.detectAsync(image, timestampMs)
    }

    private fun processResult(
        result: HandLandmarkerResult
    ) {
        val output = FloatArray(126)

        val landmarks = result.landmarks()
        val handedness = result.handedness()

        for (i in landmarks.indices) {

            if (i >= handedness.size) continue

            val label =
                handedness[i][0].categoryName()

            val handOffset =
                if (label.equals("Left", ignoreCase = true)) {
                    0
                } else {
                    63
                }

            val hand = landmarks[i]

            if (hand.size < 21) continue

            val wrist = hand[0]

            var maxDistance = 0.0

            for (landmark in hand) {
                val dx = landmark.x() - wrist.x()
                val dy = landmark.y() - wrist.y()
                val dz = landmark.z() - wrist.z()

                val distance =
                    sqrt(
                        (dx * dx +
                                dy * dy +
                                dz * dz).toDouble()
                    )

                if (distance > maxDistance) {
                    maxDistance = distance
                }
            }

            if (maxDistance == 0.0) continue

            for (j in 0 until 21) {

                val landmark = hand[j]

                val x =
                    ((landmark.x() - wrist.x()) / maxDistance)
                        .toFloat()

                val y =
                    ((landmark.y() - wrist.y()) / maxDistance)
                        .toFloat()

                val z =
                    ((landmark.z() - wrist.z()) / maxDistance)
                        .toFloat()

                val index =
                    handOffset + j * 3

                output[index] = x
                output[index + 1] = y
                output[index + 2] = z
            }
        }

        onResult(output)
    }

    fun close() {
        handLandmarker.close()
    }
}