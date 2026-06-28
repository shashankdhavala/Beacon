package com.beacon.workermvp

import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.security.MessageDigest

data class TensorPayload(
    val messageType: String,
    val requestId: String,
    val step: Int,
    val sourceShard: Int,
    val targetShard: Int,
    val shape: IntArray,
    val dtype: String,
    val bytes: ByteArray,
    val responseMode: String = "echo",
    val includeChecksum: Boolean = true,
    val route: String = "",
    val currentLength: Int = 0,
    val modelId: String = "",
) {
    fun toHeaderJson(): JSONObject {
        return JSONObject()
            .put("messageType", messageType)
            .put("requestId", requestId)
            .put("step", step)
            .put("sourceShard", sourceShard)
            .put("targetShard", targetShard)
            .put("shape", JSONArray(shape.toList()))
            .put("dtype", dtype)
            .put("byteLength", bytes.size)
            .put("sha256", if (includeChecksum) sha256Hex(bytes) else "")
            .put("responseMode", responseMode)
            .put("route", route)
            .put("currentLength", currentLength)
            .put("modelId", modelId)
            .put("createdAtMs", System.currentTimeMillis())
    }

    fun summary(): String {
        val checksum = if (includeChecksum) sha256Hex(bytes).take(12) else "disabled"
        return "type=$messageType request=$requestId step=$step " +
            "source=$sourceShard target=$targetShard shape=${shape.contentToString()} " +
            "dtype=$dtype bytes=${bytes.size} sha256=$checksum responseMode=$responseMode " +
            "currentLength=$currentLength routeBytes=${route.length}"
    }
}

object TensorProtocol {
    fun read(input: DataInputStream): TensorPayload {
        val headerLength = input.readInt()
        require(headerLength in 1..1_000_000) { "invalid header length: $headerLength" }

        val headerBytes = ByteArray(headerLength)
        input.readFully(headerBytes)
        val header = JSONObject(String(headerBytes, Charsets.UTF_8))

        val byteLength = header.getInt("byteLength")
        require(byteLength >= 0) { "invalid body length: $byteLength" }

        val body = ByteArray(byteLength)
        input.readFully(body)

        val expectedSha = header.optString("sha256", "")
        if (expectedSha.isNotEmpty()) {
            val actualSha = sha256Hex(body)
            require(expectedSha == actualSha) {
                "checksum mismatch expected=$expectedSha actual=$actualSha"
            }
        }

        return TensorPayload(
            messageType = header.getString("messageType"),
            requestId = header.getString("requestId"),
            step = header.getInt("step"),
            sourceShard = header.getInt("sourceShard"),
            targetShard = header.getInt("targetShard"),
            shape = header.getJSONArray("shape").toIntArray(),
            dtype = header.getString("dtype"),
            bytes = body,
            responseMode = header.optString("responseMode", "echo"),
            includeChecksum = expectedSha.isNotEmpty(),
            route = header.optString("route", ""),
            currentLength = header.optInt("currentLength", 0),
            modelId = header.optString("modelId", ""),
        )
    }

    fun write(output: DataOutputStream, payload: TensorPayload) {
        val headerBytes = payload.toHeaderJson().toString().toByteArray(Charsets.UTF_8)
        output.writeInt(headerBytes.size)
        output.write(headerBytes)
        output.write(payload.bytes)
        output.flush()
    }
}

fun JSONArray.toIntArray(): IntArray {
    return IntArray(length()) { index -> getInt(index) }
}

fun sha256Hex(bytes: ByteArray): String {
    val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
    return digest.joinToString(separator = "") { "%02x".format(it) }
}
