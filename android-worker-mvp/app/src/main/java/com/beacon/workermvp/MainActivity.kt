package com.beacon.workermvp

import android.app.Activity
import android.os.Bundle
import android.text.method.ScrollingMovementMethod
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import java.net.Inet4Address
import java.net.NetworkInterface

class MainActivity : Activity() {
    private lateinit var logView: TextView
    private lateinit var portInput: EditText
    private var server: ActivationServer? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 32, 32, 32)
        }

        val title = TextView(this).apply {
            text = "Beacon Worker MVP\ndiagnostic persistent TCP build"
            textSize = 24f
        }
        root.addView(title)

        val ipLabel = TextView(this).apply {
            text = "Phone IPs:\n${localIpv4AddressSummary()}"
            textSize = 18f
            setPadding(0, 24, 0, 16)
        }
        root.addView(ipLabel)

        portInput = EditText(this).apply {
            setText("9000")
            hint = "Port"
            inputType = android.text.InputType.TYPE_CLASS_NUMBER
        }
        root.addView(portInput)

        val startButton = Button(this).apply {
            text = "Start Worker Server"
            setOnClickListener {
                val port = portInput.text.toString().toIntOrNull() ?: 9000
                server?.stop()
                server = ActivationServer(port) { line -> appendLog(line) }
                server?.start()
                appendLog("Use this from Mac: python3 tools/mac_coordinator.py --host ${preferredLocalIpv4Address()} --port $port")
            }
        }
        root.addView(startButton)

        val stopButton = Button(this).apply {
            text = "Stop"
            setOnClickListener {
                server?.stop()
                server = null
                appendLog("Stop requested")
            }
        }
        root.addView(stopButton)

        logView = TextView(this).apply {
            textSize = 14f
            movementMethod = ScrollingMovementMethod()
            setPadding(0, 24, 0, 0)
        }
        root.addView(
            logView,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                0,
                1f,
            ),
        )

        setContentView(root)
    }

    override fun onDestroy() {
        server?.stop()
        super.onDestroy()
    }

    private fun appendLog(line: String) {
        runOnUiThread {
            logView.append("${System.currentTimeMillis()}  $line\n")
        }
    }

    private fun preferredLocalIpv4Address(): String? {
        return localIpv4Addresses().firstOrNull()?.address
    }

    private fun localIpv4AddressSummary(): String {
        val addresses = localIpv4Addresses()
        if (addresses.isEmpty()) return "connect to WiFi first"
        return addresses.joinToString(separator = "\n") { "${it.interfaceName}: ${it.address}" }
    }

    private fun localIpv4Addresses(): List<AddressInfo> {
        val result = mutableListOf<AddressInfo>()
        val interfaces = NetworkInterface.getNetworkInterfaces().toList()
        for (networkInterface in interfaces) {
            if (!networkInterface.isUp || networkInterface.isLoopback) continue
            val addresses = networkInterface.inetAddresses.toList()
            for (address in addresses) {
                if (address is Inet4Address && !address.isLoopbackAddress) {
                    result += AddressInfo(networkInterface.name, address.hostAddress)
                }
            }
        }
        return result.sortedWith(
            compareBy<AddressInfo> {
                when {
                    it.interfaceName == "wlan0" -> 0
                    it.interfaceName.startsWith("wlan") -> 1
                    it.interfaceName.startsWith("ap") -> 2
                    it.interfaceName.startsWith("swlan") -> 3
                    it.interfaceName.startsWith("rmnet") -> 9
                    else -> 5
                }
            }.thenBy { it.interfaceName },
        )
    }
}

data class AddressInfo(
    val interfaceName: String,
    val address: String,
)
