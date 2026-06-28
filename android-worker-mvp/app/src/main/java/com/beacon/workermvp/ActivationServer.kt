package com.beacon.workermvp

import org.json.JSONArray
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class ActivationServer(
    private val port: Int,
    private val shardId: Int,
    private val connectTimeoutMs: Int = 5_000,
    private val log: (String) -> Unit,
) {
    private val running = AtomicBoolean(false)
    private var serverSocket: ServerSocket? = null

    fun start() {
        if (!running.compareAndSet(false, true)) return

        thread(name = "activation-server", isDaemon = true) {
            try {
                ServerSocket(port).use { server ->
                    serverSocket = server
                    log("Worker listening on port $port (route-in-message mode, shard=$shardId)")

                    while (running.get()) {
                        val socket = server.accept()
                        handleClient(socket)
                    }
                }
            } catch (error: Exception) {
                if (running.get()) {
                    log("Server error: ${error.message}")
                }
            } finally {
                running.set(false)
                serverSocket = null
                log("Worker stopped")
            }
        }
    }

    fun stop() {
        running.set(false)
        try {
            serverSocket?.close()
        } catch (_: Exception) {
        }
    }

    private fun handleClient(socket: Socket) {
        thread(name = "activation-client", isDaemon = true) {
            socket.use { client ->
                client.tcpNoDelay = true
                client.keepAlive = false
                client.receiveBufferSize = 2 * 1024 * 1024
                client.sendBufferSize = 2 * 1024 * 1024
                val remote = "${client.inetAddress.hostAddress}:${client.port}"
                log("Client connected: $remote")

                try {
                    val input = DataInputStream(client.getInputStream().buffered())
                    val output = DataOutputStream(client.getOutputStream().buffered())

                    while (running.get() && !client.isClosed) {
                        val startedAt = System.nanoTime()
                        val request = try {
                            TensorProtocol.read(input)
                        } catch (_: EOFException) {
                            log("Client closed: $remote")
                            break
                        }
                        val receiveMs = (System.nanoTime() - startedAt) / 1_000_000.0
                        log("Received ${request.summary()} read_ms=${"%.2f".format(receiveMs)}")

                        val response = if (request.messageType == "TEXT") {
                            handleTextRoute(request)
                        } else {
                            // MVP behavior: echo the exact tensor bytes back as a RESULT.
                            // Later this is where ExecuTorch shard B will run.
                            request.copy(
                                messageType = "RESULT",
                                sourceShard = request.targetShard,
                                targetShard = request.sourceShard,
                                bytes = if (request.responseMode == "ack") ByteArray(0) else request.bytes,
                            )
                        }
                        TensorProtocol.write(output, response)
                        log("Sent ${response.summary()}")
                    }
                } catch (error: Exception) {
                    log("Client error: ${error.message}")
                }
            }
        }
    }

    private fun handleTextRoute(request: TensorPayload): TensorPayload {
        val incomingText = request.bytes.toString(Charsets.UTF_8)
        val appendedText = incomingText + shardId
        val appendedBytes = appendedText.toByteArray(Charsets.UTF_8)
        log("Text route: '$incomingText' -> '$appendedText'")

        val route = parseRoute(request.route)
        val currentIndex = route.indexOfFirst { it.shardId == shardId }
        val nextHop = if (currentIndex >= 0) route.getOrNull(currentIndex + 1) else null

        val appendedPayload = request.copy(
            messageType = "TEXT",
            step = request.step + 1,
            sourceShard = shardId,
            targetShard = nextHop?.shardId ?: 0,
            shape = intArrayOf(appendedBytes.size),
            dtype = "utf8",
            bytes = appendedBytes,
        )

        if (route.isEmpty()) {
            log("No route metadata; returning as terminal shard")
            return appendedPayload.copy(messageType = "RESULT")
        }

        if (currentIndex < 0) {
            log("Shard $shardId not found in route; returning as terminal shard")
            return appendedPayload.copy(messageType = "RESULT")
        }

        if (nextHop == null) {
            return appendedPayload.copy(messageType = "RESULT")
        }

        log("Forwarding text request=${request.requestId} to shard ${nextHop.shardId} at ${nextHop.host}:${nextHop.port}")
        return forwardOnce(nextHop.host, nextHop.port, appendedPayload)
    }

    private fun parseRoute(routeJson: String): List<RouteHop> {
        if (routeJson.isBlank()) return emptyList()
        val array = JSONArray(routeJson)
        return List(array.length()) { index ->
            val item = array.getJSONObject(index)
            RouteHop(
                shardId = item.getInt("shardId"),
                host = item.getString("host"),
                port = item.optInt("port", 9000),
            )
        }
    }

    private fun forwardOnce(host: String, port: Int, payload: TensorPayload): TensorPayload {
        Socket().use { socket ->
            socket.tcpNoDelay = true
            socket.keepAlive = false
            socket.receiveBufferSize = 2 * 1024 * 1024
            socket.sendBufferSize = 2 * 1024 * 1024
            socket.connect(java.net.InetSocketAddress(host, port), connectTimeoutMs)
            socket.soTimeout = connectTimeoutMs

            val input = DataInputStream(socket.getInputStream().buffered())
            val output = DataOutputStream(socket.getOutputStream().buffered())
            TensorProtocol.write(output, payload)
            val response = TensorProtocol.read(input)
            log("Forward response ${response.summary()}")
            return response
        }
    }
}

data class RouteHop(
    val shardId: Int,
    val host: String,
    val port: Int,
)
