import time
import pyRAPL
import socket
import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

# Setup PyRAPL
pyRAPL.setup()

def establish_socket(role, target_ip="127.0.0.1", port=5000):
    """
    Establish a persistent socket connection.
    Args:
        role (str): Role in communication ('sender' or 'receiver').
        target_ip (str): Target IP address for the receiver. Defaults to localhost.
        port (int): Port for the connection. Defaults to 5000.

    Returns:
        socket: Established socket object.
    """
    if role == "sender":
        sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        retry_delay = 0.01
        while 1:
            try:
                sender_socket.connect((target_ip, port))
                print("Sender connected to receiver.")
                return sender_socket
            except (socket.timeout, ConnectionRefusedError) as e:
                time.sleep(retry_delay)
        # print("Sender connected to receiver.")
        # return sender_socket

    elif role == "receiver":
        receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        receiver_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        receiver_socket.bind(("0.0.0.0", port))
        receiver_socket.listen(60)
        print("Receiver is listening...")
        conn, addr = receiver_socket.accept()
        print(f"Receiver connected by {addr}.")
        return conn

    else:
        raise ValueError("Invalid role specified. Use 'sender' or 'receiver'.")

def simulate_communication(phase_name, duration, bandwidth, packet_size, socket_obj, role, data_chunk):
    """
    Simulate communication workload (upload/download) using an established socket.
    Args:
        phase_name (str): Name of the phase (e.g., "Upload 100MB/s").
        duration (int): Duration of the experiment in seconds.
        bandwidth (int): Bandwidth in bytes per second.
        packet_size (int): network packet size (MTU) in bytes.
        socket_obj (socket): Established socket object.
        role (str): Role in communication ('sender' or 'receiver').
    """
    if bandwidth<=300 * 1024**2:
        total = 1000 * 1024**2
    else:
        total = 10000 * 1024**2
    print(f"{phase_name} started as {role}.")
    if bandwidth < 100000:
        time.sleep(duration)
        return 0
    
    os.system(f"sudo tc qdisc add dev lo root netem rate {bandwidth * 8}bit")
    # print("packetsize:",packet_size)
    os.system(f"sudo ip link set dev enp4s0 mtu {packet_size}")
    if role == "sender":
        data_transmitted = 0
        lenpack = len(data_chunk)
        # start_time = time.time()
        while data_transmitted < total:
            try:
                socket_obj.sendall(data_chunk)
                data_transmitted += lenpack
            except ConnectionResetError:
                print("Connection reset by receiver. Ending transmission.")

        print(f"{phase_name} completed. Data transmitted: {data_transmitted / 1e6:.2f} MB")

    elif role == "receiver":
        data_received = 0
        # start_time = time.time()
        # print(total/packet_size)
        while data_received < total:
            try:
                # data = socket_obj.recv(packet_size)65483
                data = socket_obj.recv(65483)
                data_received += len(data)
                # print(len(data))
            except ConnectionResetError:
                print("Connection reset by sender. Ending reception.")
                break

        print(f"{phase_name} completed. Data received: {data_received / 1e6:.2f} MB")
        # data_transmitted = data_received

    os.system(f"sudo tc qdisc del dev lo root netem rate {bandwidth * 8}bit")
    os.system(f"sudo ip link set dev enp4s0 mtu 9001")
    return total

def run_experiment(name, duration, bandwidth, packet_size, role, socket_obj, data_chunk):
    """
    Run a single communication experiment.
    Args:
        name (str): Name of the experiment.
        duration (int): Duration of the experiment.
        bandwidth (int): Bandwidth in bytes per second.
        packet_size (int): network packet size (MTU) in bytes.
        role (str): Role in communication ('sender' or 'receiver').
        socket_obj (socket): Established socket object.
    """
    
    # time.sleep(1)
    if role == "sender":
        # # Sender sends "READY"
        # socket_obj.sendall(b"READY")
        # # Expect "OK" from receiver
        resp = socket_obj.recv(5)
        if resp != b"OK":
            print("Synchronization error: Did not receive OK from receiver.")
    else:
        # Receiver waits for "READY"
        # data = socket_obj.recv(5)
        # if data != b"READY":
        #     print("Synchronization error: Did not receive READY from sender.")
        # Respond with "OK"
        socket_obj.sendall(b"OK")
    print(f"\nStarting experiment: {name}, network packet size (MTU): {packet_size} bytes")

    meter = pyRAPL.Measurement('transmission_power')
    meter.begin()
    start_time = time.time()

    data_transmitted = simulate_communication(name, duration, bandwidth, packet_size, socket_obj, role, data_chunk)

    meter.end()
    end_time = time.time()
    elapsed_time = end_time - start_time #duration
    energy_used = meter.result.pkg[0] / 1e6  - 17.5*elapsed_time # Convert µJ to J
    avg_power = energy_used / elapsed_time if elapsed_time > 0 else 0
    speed = data_transmitted / elapsed_time / 1024**2  # MB/s
    energy_per_mb = energy_used / (data_transmitted / 1024**2) if data_transmitted > 0 else 0

    print(f"\nExperiment {name} results:")
    print(f"Energy Used: {energy_used:.2f} Joules")
    print(f"Avg Power: {avg_power:.2f} Watts")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    print(f"Speed of transmission: {speed:.2f} MB/sec")
    print(f"Energy per MB: {energy_per_mb:.2f} J/MB")

    return energy_used, avg_power, speed, energy_per_mb

def main():
    parser = argparse.ArgumentParser(description="Simulate data communication workload.")
    parser.add_argument("--role", choices=["sender", "receiver"], required=True, help="Role of the machine: sender or receiver.")
    args = parser.parse_args()
    # Experiment configurations
    experiments = [
        {"name": "Experiment 0", "bandwidth": 0},
        {"name": "Experiment 1", "bandwidth": 100 * 1024**2},  # 100MB/s
        {"name": "Experiment 2", "bandwidth": 300 * 1024**2},  # 300MB/s
        {"name": "Experiment 3", "bandwidth": 1000 * 1024**2},  # 600MB/s
        {"name": "Experiment 4", "bandwidth": 3000 * 1024**2},  # 600MB/s
    ]

    # packet_sizes = [4092, 16371, 65483 , 261936, 1024**2, 4*1024**2]  #   1KB, 1MB, 10MB, 100MB
    packet_sizes = [1500, 3000, 6000, 9000]
    duration = 5  # Duration for each experiment in seconds
    target_ip = "172.31.44.82"  # Replace with actual IP of the other machine
    port = 100

    role = args.role
    
    results = []
    
    for exp in experiments:
        for packet_size in packet_sizes:
            if exp["bandwidth"]<=300 * 1024**2:
                total = 1000 * 1024**2
            else:
                total = 10000 * 1024**2
            data_chunk = os.urandom(int(total/1000))  # Generate random data
            socket_obj = establish_socket(role, target_ip, port)
            # socket_obj.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, int(packet_size*1.1))
            result = run_experiment(exp["name"], duration, exp["bandwidth"], packet_size, role, socket_obj, data_chunk)
            results.append((exp["name"], exp["bandwidth"], packet_size, *result))
            socket_obj.close()

        # Organize results
    energy_per_mb_data = {bandwidth: [] for bandwidth in [exp["bandwidth"] for exp in experiments]}
    power_data = {packet_size: [] for packet_size in packet_sizes}
    data_speed_data = {packet_size: [] for packet_size in packet_sizes}

    for res in results:
        _, bandwidth, packet_size, _, avg_power, speed, e_per_mb = res
        energy_per_mb_data[bandwidth].append(e_per_mb)
        power_data[packet_size].append(avg_power)
        data_speed_data[packet_size].append(speed)
        if bandwidth == 0:
            average_idle_power = avg_power
    
    # Plot energy/MB vs network packet size (MTU)
    plt.figure(figsize=(8, 6))
    for bandwidth, values in energy_per_mb_data.items():
        if bandwidth==0:
            continue
        plt.plot([ps / 1024 for ps in packet_sizes], values, marker='o', linestyle='-', linewidth=2, markersize=8, label=f"Bit Rate: {bandwidth / 1024**2} MB/s")
    plt.title("Active Energy/MB vs network packet size (MTU)", fontsize=16)
    plt.xlabel("network packet size (MTU) (KB)", fontsize=14)
    plt.ylabel("Active Energy (J/MB)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("energy_per_mb_vs_packet_size_{}.png".format(role))

    # Plot energy/MB vs bit rate
    plt.figure(figsize=(8, 6))
    for i,packet_size in enumerate(packet_sizes):
        # print([exp["bandwidth"] / 1024**2 for exp in experiments])
        # print(energy_per_mb_data,[exp["bandwidth"] for exp in experiments])
        # print(i)
        # print([energy_per_mb_data[bw][i] for i, bw in enumerate([exp["bandwidth"] for exp in experiments])])
        plt.plot([exp["bandwidth"] / 1024**2 for exp in experiments[1:]], [energy_per_mb_data[bw][i] for bw in [exp["bandwidth"] for exp in experiments[1:]]], marker='o', linestyle='-', linewidth=2, markersize=8, label=f"network packet size (MTU): {packet_size / 1024**2} MB")
    plt.title("Active Energy/MB vs Bit Rate", fontsize=16)
    plt.xlabel("Max Bit Rate (MB/s)", fontsize=14)
    plt.ylabel("Active Energy (J/MB)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("energy_per_mb_vs_bit_rate_{}.png".format(role))
    
    
    
    # Plot energy/MB vs bit rate log-log scale
    plt.figure(figsize=(8, 6))
    for i, packet_size in enumerate(packet_sizes):
        x_values = [exp["bandwidth"] / 1024**2 for exp in experiments[1:]]
        y_values = [energy_per_mb_data[bw][i] for bw in [exp["bandwidth"] for exp in experiments[1:]]]
        
        # Plotting with log-transformed data
        plt.plot(
            np.log10(x_values),  # Log-transformed x-axis
            np.log10(y_values),  # Log-transformed y-axis
            marker='o', linestyle='-', linewidth=2, markersize=8,
            label=f"Network packet size (MTU): {packet_size / 1024**2:.3f} MB"
        )

    # Custom ticks for the log-log plot
    x_ticks = np.log10([exp["bandwidth"] / 1024**2 for exp in experiments[1:]])
    x_tick_labels = [f"{bw / 1024**2:.1f}" for bw in [exp["bandwidth"] for exp in experiments[1:]]]
    y_ticks = np.log10([min(y_values), max(y_values)])
    y_tick_labels = [f"{val:.2f}" for val in [min(y_values), max(y_values)]]

    plt.xticks(ticks=x_ticks, labels=x_tick_labels, fontsize=12)
    plt.yticks(ticks=y_ticks, labels=y_tick_labels, fontsize=12)

    # Labels and Title
    plt.title("Active Energy/MB vs Bit Rate (Log–log Scale)", fontsize=16)
    plt.xlabel("Max Bit Rate (MB/s)", fontsize=14)
    plt.ylabel("Active Energy (J/MB)", fontsize=14)

    # Grid, legend, and layout adjustments
    plt.grid(True, linestyle='--', alpha=0.7, which="both")
    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save the plot (optional)
    plt.savefig("loglog_energy_per_mb_vs_bit_rat_{}.png".format(role))



    # Plot power vs Max Bit Rate
    plt.figure(figsize=(8, 6))
    for packet_size, values in power_data.items():
        plt.plot(data_speed_data[packet_size], values, marker='o', linestyle='-', linewidth=2, markersize=8, label=f"network packet size (MTU): {packet_size / 1024} KB")
    plt.title("Active Power Consumption vs Max Bit Rate", fontsize=16)
    plt.xlabel("Max Bit Rate (MB/s)", fontsize=14)
    plt.ylabel("Active Power (Watts)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("power_vs_speed_{}.png".format(role))

    
    plt.figure(figsize=(8, 6))
    for packet_size in packet_sizes:
        actual_speeds = [data_speed_data[packet_size][i] for i, bw in enumerate([exp["bandwidth"] for exp in experiments])]
        plt.plot([exp["bandwidth"] / 1024**2 for exp in experiments], actual_speeds, marker='o', linestyle='-', linewidth=2, markersize=8, label=f"network packet size (MTU): {packet_size / 1024} KB")
    plt.title("Actual Transmitted Bit Rate vs Max Bit Rateped Bit Rate", fontsize=16)
    plt.xlabel("Max Bit Rate (MB/s)", fontsize=14)
    plt.ylabel("Actual Transmitted Bit Rate (MB/s)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig("actual_vs_Max_bit_rate_{}.png".format(role))
    

if __name__ == "__main__":
    main()