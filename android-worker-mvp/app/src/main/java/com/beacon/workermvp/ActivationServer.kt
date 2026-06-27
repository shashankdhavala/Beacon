package com.beacon.workermvp

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class ActivationServer(
    private val port: Int,
    private val appendSuffix: String,
    private val nextHost: String?,
    private val nextPort: Int,
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
                    val route = if (nextHost.isNullOrBlank()) {
                        "terminal node"
                    } else {
                        "forwarding to $nextHost:$nextPort"
                    }
                    log("Worker listening on port $port (diagnostic persistent mode, append='$appendSuffix', $route)")

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
        val appendedText = incomingText + appendSuffix
        val appendedBytes = appendedText.toByteArray(Charsets.UTF_8)
        log("Text route: '$incomingText' -> '$appendedText'")

        val appendedPayload = request.copy(
            messageType = "TEXT",
            step = request.step + 1,
            sourceShard = request.targetShard,
            targetShard = request.targetShard + 1,
            shape = intArrayOf(appendedBytes.size),
            dtype = "utf8",
            bytes = appendedBytes,
        )

        val host = nextHost?.takeIf { it.isNotBlank() }
        if (host == null) {
            return appendedPayload.copy(messageType = "RESULT")
        }

        log("Forwarding text request=${request.requestId} to $host:$nextPort")
        return forwardOnce(host, nextPort, appendedPayload)
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
