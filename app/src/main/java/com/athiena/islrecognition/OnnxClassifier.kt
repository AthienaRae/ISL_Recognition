package com.athiena.islrecognition

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

class OnnxClassifier(context: Context) {

    private val environment = OrtEnvironment.getEnvironment()

    private val session: OrtSession

    private val labels = arrayOf(
        "Hello",
        "How are you",
        "Alright",
        "Good Morning",
        "Good afternoon",
        "Good evening",
        "Good night",
        "Thank you",
        "Pleased"
    )

    init {
        val modelBytes =
            context.assets
                .open("gru_greetings.onnx")
                .readBytes()

        session = environment.createSession(
            modelBytes,
            OrtSession.SessionOptions()
        )
    }

    fun predict(frames: List<FloatArray>): Prediction {

        require(frames.size == 91) {
            "Expected 91 frames, got ${frames.size}"
        }

        val inputData = FloatArray(91 * 126)

        var index = 0

        for (frame in frames) {
            require(frame.size == 126) {
                "Expected 126 features, got ${frame.size}"
            }

            for (value in frame) {
                inputData[index++] = value
            }
        }

        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(inputData),
            longArrayOf(1, 91, 126)
        )

        inputTensor.use {

            val inputName =
                session.inputNames.iterator().next()

            val results =
                session.run(
                    mapOf(
                        inputName to inputTensor
                    )
                )

            results.use {

                val output =
                    it[0].value as Array<FloatArray>

                val logits = output[0]

                val probabilities = FloatArray(logits.size)

                var maxLogit = logits[0]

                for (i in 1 until logits.size) {
                    if (logits[i] > maxLogit) {
                        maxLogit = logits[i]
                    }
                }

                var sum = 0.0

                for (i in logits.indices) {
                    probabilities[i] =
                        kotlin.math.exp((logits[i] - maxLogit).toDouble()).toFloat()

                    sum += probabilities[i]
                }

                for (i in probabilities.indices) {
                    probabilities[i] =
                        (probabilities[i] / sum).toFloat()
                }

                var bestIndex = 0
                var bestScore = probabilities[0]

                for (i in 1 until probabilities.size) {
                    if (probabilities[i] > bestScore) {
                        bestScore = probabilities[i]
                        bestIndex = i
                    }
                }

                return Prediction(
                    label = labels[bestIndex],
                    confidence = bestScore
                )
            }
        }
    }

    fun close() {
        session.close()
    }

    data class Prediction(
        val label: String,
        val confidence: Float
    )
}