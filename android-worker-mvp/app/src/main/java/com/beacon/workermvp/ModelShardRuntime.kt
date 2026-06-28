package com.beacon.workermvp

import android.content.Context
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ModelShardRuntime(
    private val context: Context,
    private val assetDir: String,
    routeShardId: Int,
    private val log: (String) -> Unit,
) {
    private val manifest = ModelManifest.load(context, assetDir)
    private val manifestShardIndex = routeShardId - 1
    private val shard = manifest.shards.firstOrNull { it.shardIndex == manifestShardIndex }
        ?: error("No shard artifact for route shard id $routeShardId")
    private val module: Module
    private var kvCache: List<Tensor>? = null

    init {
        val modelPath = copyAssetToFilesDir("${assetDir}/${shard.artifactPath}")
        module = Module.load(modelPath.absolutePath)
        log(
            "Loaded ${manifest.modelId} shard route=$routeShardId manifest=${shard.shardIndex} " +
                "layers=${shard.startLayer}-${shard.endLayer} path=${modelPath.name}",
        )
    }

    fun startRequest() {
        kvCache = allocateCache()
        log("Model request started; local cache tensors=${kvCache?.size ?: 0}")
    }

    fun flushRequest() {
        kvCache = null
        log("Model request flushed")
    }

    fun runStep(request: TensorPayload): TensorPayload {
        if (request.dtype != "float32") {
            error("STEP_HIDDEN requires dtype=float32, got ${request.dtype}")
        }
        if (request.currentLength <= 0) {
            error("STEP_HIDDEN requires currentLength > 0")
        }
        if (kvCache == null) {
            startRequest()
        }

        val inputFloats = request.bytes.toFloatArrayLe()
        val hidden = Tensor.fromBlob(inputFloats, request.shape.toLongShape())
        val attentionMask = Tensor.fromBlob(
            LongArray(manifest.maxCacheLen) { index -> if (index < request.currentLength) 1L else 0L },
            longArrayOf(1L, manifest.maxCacheLen.toLong()),
        )
        val cachePosition = Tensor.fromBlob(longArrayOf(request.step.toLong()), longArrayOf(1L))
        val inputs = mutableListOf(
            EValue.from(hidden),
            EValue.from(attentionMask),
            EValue.from(cachePosition),
        )
        for (cacheTensor in kvCache.orEmpty()) {
            inputs += EValue.from(cacheTensor)
        }

        val startedAt = System.nanoTime()
        val outputs = try {
            module.forward(*inputs.toTypedArray())
        } catch (error: Exception) {
            val nativeLogs = module.readLogBuffer().joinToString(separator = "\n")
            val detail = buildString {
                append("ExecuTorch forward failed on route shard ")
                append(shard.shardIndex + 1)
                append(" manifest shard ")
                append(shard.shardIndex)
                append(": ")
                append(error.message)
                append("; inputs hidden=")
                append(request.shape.contentToString())
                append(" bytes=")
                append(request.bytes.size)
                append(" step=")
                append(request.step)
                append(" currentLength=")
                append(request.currentLength)
                append(" cacheTensors=")
                append(kvCache.orEmpty().size)
                if (nativeLogs.isNotBlank()) {
                    append("; native logs:\n")
                    append(nativeLogs)
                }
            }
            throw RuntimeException(detail, error)
        }
        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000.0
        val outputTensors = outputs.map { it.toTensor() }
        val hiddenIndex = outputTensors.indexOfFirst { tensor ->
            tensor.shape().contentEquals(request.shape.toLongShape())
        }.takeIf { it >= 0 } ?: outputTensors.indexOfFirst { tensor ->
            tensor.shape().size == 3
        }
        if (hiddenIndex < 0) {
            error(
                "ExecuTorch forward did not return a rank-3 hidden tensor; " +
                    "outputShapes=${outputTensors.map { it.shape().contentToString() }}",
            )
        }

        val outputTensor = outputTensors[hiddenIndex]
        val expectedCacheTensors = shard.numLayers * 2
        kvCache = outputTensors
            .filterIndexed { index, tensor -> index != hiddenIndex && tensor.shape().size == 4 }
            .take(expectedCacheTensors)

        val outputShape = outputTensor.shape().map { it.toInt() }.toIntArray()
        val outputBytes = outputTensor.getDataAsFloatArray().toBytesLe()
        log(
            "Ran shard=${shard.shardIndex} step=${request.step} currentLength=${request.currentLength} " +
                "in=${request.shape.contentToString()} out=${outputShape.contentToString()} " +
                "hiddenOutput=$hiddenIndex cacheOutputs=${kvCache.orEmpty().size} run_ms=${"%.2f".format(elapsedMs)}",
        )

        return request.copy(
            shape = outputShape,
            dtype = "float32",
            bytes = outputBytes,
            includeChecksum = request.includeChecksum,
        )
    }

    private fun allocateCache(): List<Tensor> {
        if (shard.numLayers == 0) return emptyList()

        val cacheShape = longArrayOf(
            1L,
            shard.numHeads.toLong(),
            manifest.maxCacheLen.toLong(),
            shard.headDim.toLong(),
        )
        val numElements = 1 * shard.numHeads * manifest.maxCacheLen * shard.headDim
        val tensors = mutableListOf<Tensor>()
        repeat(shard.numLayers) {
            tensors += Tensor.fromBlob(FloatArray(numElements), cacheShape)
            tensors += Tensor.fromBlob(FloatArray(numElements), cacheShape)
        }
        return tensors
    }

    private fun copyAssetToFilesDir(assetPath: String): File {
        val targetDir = File(context.filesDir, "model-assets/${assetDir}").apply { mkdirs() }
        val target = File(targetDir, File(assetPath).name)
        context.assets.open(assetPath).use { input ->
            target.outputStream().use { output -> input.copyTo(output) }
        }
        return target
    }
}

data class ModelManifest(
    val modelId: String,
    val maxCacheLen: Int,
    val shards: List<ModelShardInfo>,
) {
    companion object {
        fun load(context: Context, assetDir: String): ModelManifest {
            val text = context.assets.open("${assetDir}/manifest.json").bufferedReader().use { it.readText() }
            val root = JSONObject(text)
            val shardArray = root.getJSONArray("shards")
            val shards = List(shardArray.length()) { index ->
                val item = shardArray.getJSONObject(index)
                ModelShardInfo(
                    shardIndex = item.getInt("shard_index"),
                    artifactPath = item.getString("artifact_path"),
                    startLayer = item.getInt("start_layer"),
                    endLayer = item.getInt("end_layer"),
                    numLayers = item.getInt("num_layers"),
                    numHeads = item.getInt("num_heads"),
                    headDim = item.getInt("head_dim"),
                )
            }
            return ModelManifest(
                modelId = root.getString("model_id"),
                maxCacheLen = root.getInt("max_cache_len"),
                shards = shards,
            )
        }
    }
}

data class ModelShardInfo(
    val shardIndex: Int,
    val artifactPath: String,
    val startLayer: Int,
    val endLayer: Int,
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
)

private fun IntArray.toLongShape(): LongArray {
    return LongArray(size) { index -> this[index].toLong() }
}

private fun ByteArray.toFloatArrayLe(): FloatArray {
    require(size % 4 == 0) { "float32 byte payload must be divisible by 4, got $size" }
    val buffer = ByteBuffer.wrap(this).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(size / 4) { buffer.float }
}

private fun FloatArray.toBytesLe(): ByteArray {
    val buffer = ByteBuffer.allocate(size * 4).order(ByteOrder.LITTLE_ENDIAN)
    for (value in this) {
        buffer.putFloat(value)
    }
    return buffer.array()
}
