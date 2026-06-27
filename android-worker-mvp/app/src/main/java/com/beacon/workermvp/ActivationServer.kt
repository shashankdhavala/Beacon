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
                    log("Worker listening on port $port (diagnostic persistent mode)")

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

                        // MVP behavior: echo the exact tensor bytes back as a RESULT.
                        // Later this is where ExecuTorch shard B will run.
                        val response = TensorPayload(
                            messageType = "RESULT",
                            requestId = request.requestId,
                            step = request.step,
                            sourceShard = request.targetShard,
                            targetShard = request.sourceShard,
                            shape = request.shape,
                            dtype = request.dtype,
                            bytes = if (request.responseMode == "ack") ByteArray(0) else request.bytes,
                            responseMode = request.responseMode,
                            includeChecksum = request.includeChecksum,
                        )
                        TensorProtocol.write(output, response)
                        log("Sent ${response.summary()}")
                    }
                } catch (error: Exception) {
                    log("Client error: ${error.message}")
                }
            }
        }
    }
}
