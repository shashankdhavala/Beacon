package com.beacon.workermvp

import android.content.Context
import android.system.Os
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ModelShardRuntime(
    private val context: Context,
    private val modelDir: String,
    routeShardId: Int,
    private val log: (String) -> Unit,
) {
    private val modelSource: ModelSource
    private val manifest: ModelManifest
    private val shard: ModelShardInfo
    private val module: Module
    private var kvCache: List<Tensor>? = null

    init {
        log("Resolving model source: $modelDir")
        modelSource = ModelSource.resolve(context, modelDir)
        log("Reading model manifest")
        manifest = ModelManifest.load(modelSource)
        val manifestShardIndex = routeShardId - 1
        shard = manifest.shards.firstOrNull { it.shardIndex == manifestShardIndex }
            ?: error("No shard artifact for route shard id $routeShardId")
        val modelPath = modelSource.resolveArtifact(context, shard.artifactPath)
        log("Resolved artifact: ${modelPath.absolutePath}")
        if (manifest.exportBackend.startsWith("qnn")) {
            val nativeDir = modelSource.nativeLibraryDir(context, modelPath)
            log("Preparing QNN libraries from ${nativeDir.absolutePath}")
            prepareQnnNativeLibraries(context, nativeDir, log)
            log("QNN libraries prepared")
        }
        log("Loading ExecuTorch module from ${modelPath.absolutePath}")
        module = Module.load(modelPath.absolutePath)
        log(
            "Loaded ${manifest.modelId} shard route=$routeShardId manifest=${shard.shardIndex} " +
                "arch=${manifest.architecture} backend=${manifest.exportBackend} " +
                "layers=${shard.startLayer}-${shard.endLayer} path=${modelPath.absolutePath}",
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
        val positionIds = Tensor.fromBlob(longArrayOf(request.step.toLong()), longArrayOf(1L, 1L))
        val cachePosition = Tensor.fromBlob(longArrayOf(request.step.toLong()), longArrayOf(1L))
        val inputs = mutableListOf(EValue.from(hidden), EValue.from(attentionMask))
        if (manifest.architecture == "llama") {
            inputs += EValue.from(positionIds)
        }
        inputs += EValue.from(cachePosition)
        for (cacheTensor in kvCache.orEmpty()) {
            inputs += EValue.from(cacheTensor)
        }

        val startedAt = System.nanoTime()
        val outputs = try {
            module.forward(*inputs.toTypedArray())
        } catch (error: Throwable) {
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
                append(" architecture=")
                append(manifest.architecture)
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
}

data class ModelManifest(
    val modelId: String,
    val architecture: String,
    val exportBackend: String,
    val maxCacheLen: Int,
    val shards: List<ModelShardInfo>,
) {
    companion object {
        fun load(source: ModelSource): ModelManifest {
            val text = source.readManifest()
            val root = JSONObject(text)
            val shardArray = root.getJSONArray("shards")
            val shards = List(shardArray.length()) { index ->
                val item = shardArray.getJSONObject(index)
                ModelShardInfo(
                    shardIndex = item.getInt("shard_index"),
                    artifactPath = item.getString("artifact_path"),
                    architecture = item.optString("architecture", root.optString("architecture", "gpt2")),
                    exportBackend = item.optString("export_backend", root.optString("export_backend", "portable")),
                    startLayer = item.getInt("start_layer"),
                    endLayer = item.getInt("end_layer"),
                    numLayers = item.getInt("num_layers"),
                    numHeads = item.getInt("num_heads"),
                    headDim = item.getInt("head_dim"),
                )
            }
            return ModelManifest(
                modelId = root.getString("model_id"),
                architecture = root.optString("architecture", shards.firstOrNull()?.architecture ?: "gpt2"),
                exportBackend = root.optString("export_backend", shards.firstOrNull()?.exportBackend ?: "portable"),
                maxCacheLen = root.getInt("max_cache_len"),
                shards = shards,
            )
        }
    }
}

data class ModelShardInfo(
    val shardIndex: Int,
    val artifactPath: String,
    val architecture: String,
    val exportBackend: String,
    val startLayer: Int,
    val endLayer: Int,
    val numLayers: Int,
    val numHeads: Int,
    val headDim: Int,
)

sealed class ModelSource {
    abstract fun readManifest(): String
    abstract fun resolveArtifact(context: Context, artifactPath: String): File
    abstract fun nativeLibraryDir(context: Context, modelPath: File): File

    class Assets(private val context: Context, private val assetDir: String) : ModelSource() {
        override fun readManifest(): String {
            return context.assets.open("${assetDir}/manifest.json").bufferedReader().use { it.readText() }
        }

        override fun resolveArtifact(context: Context, artifactPath: String): File {
            val assetName = File(artifactPath).name
            val targetDir = File(context.filesDir, "model-assets/${assetDir}").apply { mkdirs() }
            val target = File(targetDir, assetName)
            context.assets.open("${assetDir}/${assetName}").use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            return target
        }

        override fun nativeLibraryDir(context: Context, modelPath: File): File {
            return modelPath.parentFile ?: context.filesDir
        }
    }

    class Directory(private val dir: File) : ModelSource() {
        override fun readManifest(): String {
            return File(dir, "manifest.json").readText()
        }

        override fun resolveArtifact(context: Context, artifactPath: String): File {
            val rawPath = File(artifactPath)
            if (rawPath.isAbsolute && rawPath.exists()) return rawPath
            val byPath = File(dir, artifactPath)
            if (byPath.exists()) return byPath
            val byName = File(dir, rawPath.name)
            if (byName.exists()) return byName
            error("Could not find shard artifact $artifactPath in ${dir.absolutePath}")
        }

        override fun nativeLibraryDir(context: Context, modelPath: File): File {
            return dir
        }
    }

    companion object {
        fun resolve(context: Context, modelDir: String): ModelSource {
            val trimmed = modelDir.trim()
            require(trimmed.isNotEmpty()) { "Model dir cannot be empty" }

            val explicit = File(trimmed)
            if (explicit.isAbsolute && File(explicit, "manifest.json").exists()) {
                return Directory(explicit)
            }

            val appExternal = File(context.getExternalFilesDir(null), "models/${trimmed}")
            if (File(appExternal, "manifest.json").exists()) {
                return Directory(appExternal)
            }

            val appInternal = File(context.filesDir, "models/${trimmed}")
            if (File(appInternal, "manifest.json").exists()) {
                return Directory(appInternal)
            }

            return Assets(context, trimmed)
        }
    }
}

private fun prepareQnnNativeLibraries(context: Context, dir: File, log: (String) -> Unit = {}) {
    if (!dir.exists()) {
        error("QNN native library directory does not exist: ${dir.absolutePath}")
    }

    Os.setenv("LD_LIBRARY_PATH", dir.absolutePath, true)
    Os.setenv("ADSP_LIBRARY_PATH", dir.absolutePath, true)

    val platformDependencies = listOf(
        NativeDependency("libc++.so", "/system/lib64/libc++.so", "/vendor/lib64/libc++.so"),
        NativeDependency("libbase.so", "/system/lib64/libbase.so", "/vendor/lib64/libbase.so"),
        NativeDependency("libcutils.so", "/system/lib64/libcutils.so", "/vendor/lib64/libcutils.so"),
        NativeDependency("libvndksupport.so", "/system/lib64/libvndksupport.so"),
        NativeDependency("libutils.so", "/system/lib64/libutils.so", "/vendor/lib64/libutils.so"),
        NativeDependency("libbinder.so", "/system/lib64/libbinder.so", "/vendor/lib64/libbinder.so"),
        NativeDependency("libhidlbase.so", "/system/lib64/libhidlbase.so", "/vendor/lib64/libhidlbase.so"),
        NativeDependency("libhardware.so", "/system/lib64/libhardware.so", "/vendor/lib64/libhardware.so"),
        NativeDependency("libbinder_ndk.so", "/system/lib64/libbinder_ndk.so"),
        NativeDependency("libdmabufheap.so", "/system/lib64/libdmabufheap.so", "/vendor/lib64/libdmabufheap.so"),
        NativeDependency("libvmmem.so", "/vendor/lib64/libvmmem.so"),
        NativeDependency(
            "android.hardware.common-V2-ndk.so",
            "/system/lib64/android.hardware.common-V2-ndk.so",
            "/vendor/lib64/android.hardware.common-V2-ndk.so",
        ),
        NativeDependency(
            "vendor.qti.hardware.dsp-V1-ndk.so",
            "/vendor/lib64/vendor.qti.hardware.dsp-V1-ndk.so",
        ),
        NativeDependency("libcdsprpc.so", "/vendor/lib64/libcdsprpc.so", "/system/vendor/lib64/libcdsprpc.so"),
    )

    val qnnLoadOrder = listOf(
        "libQnnSystem.so",
        "libQnnHtpPrepare.so",
        "libQnnHtpV79Stub.so",
        "libQnnHtp.so",
        "libqnn_executorch_backend.so",
    )
    val requiredRuntimeFiles = qnnLoadOrder + "libQnnHtpV79Skel.so"
    val missing = mutableListOf<String>()
    for (fileName in requiredRuntimeFiles) {
        if (!File(dir, fileName).exists()) {
            missing += fileName
        }
    }
    if (missing.isNotEmpty()) {
        error("Missing QNN runtime files in ${dir.absolutePath}: ${missing.joinToString()}")
    }

    val loadDir = File(context.filesDir, "qnn-runtime").apply { mkdirs() }
    for (fileName in requiredRuntimeFiles) {
        val source = File(dir, fileName)
        val target = File(loadDir, fileName)
        if (!target.exists() || target.length() != source.length()) {
            log("Copying QNN runtime file ${source.name} to app storage")
            source.inputStream().use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
        }
    }

    Os.setenv("LD_LIBRARY_PATH", loadDir.absolutePath, true)
    Os.setenv("ADSP_LIBRARY_PATH", loadDir.absolutePath, true)

    for (dependency in platformDependencies) {
        val library = dependency.resolve()
        if (library == null) {
            log("Platform dependency ${dependency.name} not found in expected system/vendor paths; letting linker resolve it")
            continue
        }
        log("Loading platform library ${library.absolutePath}")
        System.load(library.absolutePath)
    }

    for (libraryName in qnnLoadOrder) {
        val library = File(loadDir, libraryName)
        log("Loading QNN library ${library.absolutePath}")
        System.load(library.absolutePath)
    }
}

private data class NativeDependency(
    val name: String,
    val candidates: List<String>,
) {
    constructor(name: String, vararg candidates: String) : this(name, candidates.toList())

    fun resolve(): File? {
        return candidates.asSequence().map { File(it) }.firstOrNull { it.exists() }
    }
}

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
