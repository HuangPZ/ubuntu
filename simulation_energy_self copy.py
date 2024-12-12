import time
import pyRAPL
import socket
import os
import threading
import matplotlib.pyplot as plt
from scapy.all import Ether, IP, TCP, sendp, sniff

# Setup PyRAPL
pyRAPL.setup()

def establish_connection(target_ip="127.0.0.1", port=5000, interface="enp4s0"):
    """
    Establish a Scapy-based communication link.
    Args:
        target_ip (str): Target IP address for communication.
        port (int): Port for communication.
        interface (str): Network interface to use.

    Returns:
        dict: Configuration details for the communication.
    """
    return {"target_ip": target_ip, "port": port, "interface": interface}

def simulate_communication_sender(duration, bandwidth, package_size, config):
    """
    Simulate sender communication workload using Scapy.
    Args:
        duration (int): Duration of the experiment in seconds.
        bandwidth (int): Bandwidth in bytes per second.
        package_size (int): Package size in bit.
        config (dict): Configuration details for the communication.
    """
    interface = config["interface"]
    target_ip = config["target_ip"]
    port = config["port"]

    print(f"Sender started on {interface} targeting {target_ip}:{port}.")

    if bandwidth < 100000:
        time.sleep(duration)
        return 0

    os.system(f"sudo tc qdisc add dev {interface} root netem rate {bandwidth * 8}bit")

    data_transmitted = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            packet = Ether() / IP(src=target_ip, dst=target_ip) / TCP(sport=port, dport=port) / os.urandom(package_size)
            sendp(packet, iface=interface, verbose=0)
            data_transmitted += package_size
        except Exception as e:
            print(f"Error sending packet: {e}")
            break

    os.system(f"sudo tc qdisc del dev {interface} root netem rate {bandwidth * 8}bit")
    print(f"Sender completed. Data transmitted: {data_transmitted / 1e6:.2f} MB")
    return data_transmitted

def simulate_communication_receiver(duration, config):
    """
    Simulate receiver communication workload using Scapy.
    Args:
        duration (int): Duration of the experiment in seconds.
        config (dict): Configuration details for the communication.
    """
    interface = config["interface"]
    port = config["port"]

    print(f"Receiver started on {interface}.")

    data_received = 0

    def packet_callback(packet):
        nonlocal data_received
        if IP in packet and TCP in packet:
            data_received += len(packet[TCP].payload)

    sniff(iface=interface, filter=f"tcp and port {port}", prn=packet_callback, timeout=duration)
    print(f"Receiver completed. Data received: {data_received / 1e6:.2f} MB")
    return data_received

def run_experiment(name, duration, bandwidth, package_size):
    """
    Run a single communication experiment with sender and receiver.
    Args:
        name (str): Name of the experiment.
        duration (int): Duration of the experiment.
        bandwidth (int): Bandwidth in bytes per second.
        package_size (int): Package size in bytes.
    """
    print(f"\nStarting experiment: {name}, Package Size: {package_size} bit")
    target_ip = "172.31.43.45"  # Replace with actual IP of the other machine
    port = 100
    config = establish_connection(target_ip, port)

    meter = pyRAPL.Measurement('transmission_power')
    meter.begin()

    sender_thread = threading.Thread(target=simulate_communication_sender, args=(duration, bandwidth, package_size, config))
    receiver_thread = threading.Thread(target=simulate_communication_receiver, args=(duration, config))

    sender_thread.start()
    receiver_thread.start()

    sender_thread.join()
    receiver_thread.join()

    meter.end()

    elapsed_time = duration
    energy_used = meter.result.pkg[0] / 1e6  # Convert µJ to J
    avg_power = energy_used / elapsed_time if elapsed_time > 0 else 0

    print(f"\nExperiment {name} results:")
    print(f"Energy Used: {energy_used:.2f} Joules")
    print(f"Avg Power: {avg_power:.2f} Watts")

    return energy_used, avg_power

def main():
    # Experiment configurations
    experiments = [
        {"name": "Experiment 1", "bandwidth": 100 * 1024**2},  # 100MB/s
        {"name": "Experiment 2", "bandwidth": 300 * 1024**2},  # 300MB/s
        {"name": "Experiment 3", "bandwidth": 600 * 1024**2},  # 600MB/s
    ]

    package_sizes = [500, 1000, 2000, 4000, 8000]  # 1KB, 1MB, 10MB, 100MB
    duration = 5  # Duration for each experiment in seconds

    results = []

    for exp in experiments:
        for package_size in package_sizes:
            result = run_experiment(exp["name"], duration, exp["bandwidth"], package_size)
            results.append((exp["name"], exp["bandwidth"], package_size, *result))

        # Organize results
    energy_per_mb_data = {bandwidth: [] for bandwidth in [exp["bandwidth"] for exp in experiments]}
    power_data = {package_size: [] for package_size in package_sizes}
    data_speed_data = {package_size: [] for package_size in package_sizes}

    for res in results:
        _, bandwidth, package_size, _, avg_power, speed, e_per_mb = res
        energy_per_mb_data[bandwidth].append(e_per_mb)
        power_data[package_size].append(avg_power)
        data_speed_data[package_size].append(speed)

    # Plot energy/MB vs package size
    plt.figure(figsize=(8, 6))
    for bandwidth, values in energy_per_mb_data.items():
        plt.plot([ps / 1024**2 for ps in package_sizes], values, marker='o', linestyle='-', linewidth=2, markersize=8, label=f"Bit Rate: {bandwidth / 1024**2} MB/s")
    plt.title("Energy/MB vs Package Size", fontsize=16)
    plt.xlabel("Package Size (MB)", fontsize=14)
    plt.ylabel("Energy (J/MB)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("energy_per_mb_vs_package_size_{}.png".format(role))

    # Plot energy/MB vs bit rate
    plt.figure(figsize=(8, 6))
    for package_size in package_sizes:
        plt.plot([exp["bandwidth"] / 1024**2 for exp in experiments], [energy_per_mb_data[bw][i] for i, bw in enumerate([exp["bandwidth"] for exp in experiments])], marker='o', linestyle='-', linewidth=2, markersize=8, label=f"Package Size: {package_size} byte")
    plt.title("Energy/MB vs Bit Rate", fontsize=16)
    plt.xlabel("Bit Rate (MB/s)", fontsize=14)
    plt.ylabel("Energy (J/MB)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("energy_per_mb_vs_bit_rate_{}.png".format(role))

    # Plot power vs Max Bit Rate
    plt.figure(figsize=(8, 6))
    for package_size, values in power_data.items():
        plt.plot(data_speed_data[package_size], values, marker='o', linestyle='-', linewidth=2, markersize=8, label=f"Package Size: {package_size} bytes")
    plt.title("Power Consumption vs Max Bit Rate", fontsize=16)
    plt.xlabel("Max Bit Rate (MB/s)", fontsize=14)
    plt.ylabel("Power (Watts)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("power_vs_speed_{}.png".format(role))

    
    plt.figure(figsize=(8, 6))
    for package_size in package_sizes:
        actual_speeds = [data_speed_data[package_size][i] for i, bw in enumerate([exp["bandwidth"] for exp in experiments])]
        plt.plot([exp["bandwidth"] / 1024**2 for exp in experiments], actual_speeds, marker='o', linestyle='-', linewidth=2, markersize=8, label=f"Package Size: {package_size} bytes")
    plt.title("Actual Transmitted Bit Rate vs Max Bit Rateped Bit Rate", fontsize=16)
    plt.xlabel("Max Bit Rate (MB/s)", fontsize=14)
    plt.ylabel("Actual Transmitted Bit Rate (MB/s)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("actual_vs_Max_bit_rate_{}.png".format(role))
    # socket_obj.close()

if __name__ == "__main__":
    main()