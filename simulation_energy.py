import threading
import time
import psutil
import os
import pyRAPL
from cryptography.fernet import Fernet
import socket
# from socket_throttle import ThrottledSocket

# Setup PyRAPL
pyRAPL.setup()


def bind_to_cpu(core_id):
    """Bind the current process to a specific CPU core."""
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        print(f"Unable to bind to CPU core {core_id}. Ignoring.")


# def simulate_communication(phase_name, duration, bandwidth, core_id, encrypt=False):
#     """
#     Simulate communication workload (upload/download) with optional encryption.
#     Args:
#         phase_name (str): Name of the phase (e.g., "Upload 100MB/s").
#         duration (int): Duration of the experiment in seconds.
#         bandwidth (int): Bandwidth in bytes per second.
#         core_id (int): CPU core to bind the process.
#         encrypt (bool): Whether to simulate encryption/decryption.
#     """
#     print(f"{phase_name} started on CPU core {core_id}.")

#     # Bind to the specified core
#     bind_to_cpu(core_id)

#     # Simulate encryption/decryption setup
#     key = Fernet.generate_key()
#     cipher = Fernet(key)

#     # Simulate data transmission
#     data_transmitted = 0
#     chunk_size = 10 * 1024 * 1024  # 10 MB
#     start_time = time.time()

#     while time.time() - start_time < duration:
#         data_chunk = os.urandom(chunk_size)  # Generate random data
#         if encrypt:
#             encrypted_chunk = cipher.encrypt(data_chunk)  # Encrypt data
#             decrypted_chunk = cipher.decrypt(encrypted_chunk)  # Decrypt data
#         data_transmitted += chunk_size

#     # Ensure total data doesn't exceed bandwidth * duration
#     data_transmitted = min(data_transmitted, bandwidth * duration)

#     print(f"{phase_name} completed. Data transmitted: {data_transmitted / 1e6:.2f} MB")
#     return data_transmitted

def simulate_communication(phase_name, duration, bandwidth, core_id, role, target_ip="127.0.0.1", port=5000):
    """
    Simulate communication workload (upload/download) using real sockets.
    Args:
        phase_name (str): Name of the phase (e.g., "Upload 100MB/s").
        duration (int): Duration of the experiment in seconds.
        bandwidth (int): Bandwidth in bytes per second.
        core_id (int): CPU core to bind the process.
        role (str): Role in communication ('sender' or 'receiver').
        target_ip (str): Target IP address for the receiver. Defaults to localhost.
        port (int): Port for the connection. Defaults to 5000.
    """
    print(f"{phase_name} started on CPU core {core_id} as {role}.")

    # Bind to the specified core
    bind_to_cpu(core_id)

    if role == "sender":
        # Simulate data transmission as a sender
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sender_socket:
            sender_socket.connect((target_ip, port))
            data_transmitted = 0
            chunk_size = 100 * 1024 * 1024  # 10 MB
            data_chunk = os.urandom(chunk_size)  # Generate random data
            start_time = time.time()
            time_window = 1  # Time window in seconds for throttling
            bytes_per_window = bandwidth * time_window  # Convert bandwidth from MB/s to bytes per time window

            while time.time() - start_time < duration:
                window_start = time.time()
                bytes_sent_this_window = 0

                while time.time() - window_start < time_window:
                    if bandwidth == 0:
                        break
                    
                    #     # Limit the chunk size to the remaining bytes allowed in this window
                    #     chunk_size = int(bytes_per_window - bytes_sent_this_window)
                        

                    # if chunk_size <= 0:
                    #     # Wait for the next time window if the limit is reached
                    #     break

                    
                    try:
                        sender_socket.sendall(data_chunk)
                        data_transmitted += chunk_size
                        print(f"Data transmitted: {data_transmitted / 1e6:.2f} MB, Time: {time.time() - start_time:.2f} s")
                        bytes_sent_this_window += chunk_size
                    except ConnectionResetError:
                        print("Connection reset by receiver. Ending transmission.")
                        print(f"{phase_name} completed. Data transmitted: {data_transmitted / 1e6:.2f} MB")
                        return
                    # data_transmitted += chunk_size
                    # bytes_sent_this_window += chunk_size
                    if bytes_sent_this_window + chunk_size > bytes_per_window:
                        break

                # Sleep if we reach the limit before the end of the time window
                print(f"Sleeping for {max(0, time_window - (time.time() - window_start)):.2f} s")
                time.sleep(max(0, time_window - (time.time() - window_start)))

            print(f"{phase_name} completed. Data transmitted: {data_transmitted / 1e6:.2f} MB")


    elif role == "receiver":
        # Simulate data reception as a receiver
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as receiver_socket:
            try:
                receiver_socket.bind((target_ip, port))
                receiver_socket.listen(1)
                conn, addr = receiver_socket.accept()
                with conn:
                    data_received = 0
                    start_time = time.time()
                    while time.time() - start_time < duration:
                        try:
                            data = conn.recv(10 * 1024 * 1024)  # Receive data in chunks
                            if not data:
                                break
                            data_received += len(data)
                        except ConnectionResetError:
                            print("Connection reset by sender. Ending reception.")
                            break

                    print(f"{phase_name} completed. Data received: {data_received / 1e6:.2f} MB")
            except Exception as e:
                print(f"Receiver encountered an error: {e}")
    else:
        raise ValueError("Invalid role specified. Use 'sender' or 'receiver'.")

def measure_energy():
    """Measures energy using RAPL with pyRAPL."""
    meter = pyRAPL.Measurement('transmission_power')
    meter.begin()

    start_time = time.time()
    # Placeholder for the operation; this will be replaced by actual processing.
    time.sleep(1)
    end_time = time.time()

    meter.end()

    # Compute energy and elapsed time
    energy_consumed = meter.result.pkg[0] / 1e6  # Convert µJ to J
    elapsed_time = end_time - start_time

    return energy_consumed, elapsed_time


def run_experiment(name, duration, bandwidth, core_0, core_1):
    meter = pyRAPL.Measurement('transmission_power')
    meter.begin()
    
    core_0, core_1 = 0, 1  # Assign cores for sender and receiver
    port = 5000
    """
    Run a single experiment with two threads:
    - One thread for upload (on core_0).
    - One thread for download (on core_1).
    """
    print(f"\nStarting experiment: {name}")

    # Record energy before the experiment
    energy_before, _ = measure_energy()
    start_time = time.time()
    # energy_start = meter.result.pkg[0] / 1e6  # Convert µJ to J
    # Threads for communication
   
    download_thread =  threading.Thread(
        target=simulate_communication,
        args=("Download", duration, bandwidth, core_1, "receiver", "127.0.0.1", port),
    )
    
    upload_thread = threading.Thread(
        target=simulate_communication,
        args=("Upload", duration, bandwidth, core_0, "sender", "127.0.0.1", port),
    )
    
    # Start threads
    print(f"Starting threads on download.")
    download_thread.start()
    print(f"Starting threads on upload.")
    upload_thread.start()
    

    # Wait for threads to complete
    upload_thread.join()
    download_thread.join()

    # Record energy after the experiment
    meter.end()
    # energy_end = meter.result.pkg[0] / 1e6  # Convert µJ to J
    end_time = time.time()
    elapsed_time = end_time - start_time
    energy_used = meter.result.pkg[0] / 1e6  # Convert µJ to J

    avg_power = energy_used / elapsed_time if elapsed_time > 0 else 0

    print(f"\nExperiment {name} results:")
    print(f"Energy Used: {energy_used:.2f} Joules")
    print(f"Avg Power: {avg_power:.2f} Watts\n")
    print(f"Elapsed Time: {elapsed_time:.2f} seconds")

    return energy_used, avg_power


def main():
    # Experiment configurations
    experiments = [
        {"name": "Experiment 5", "bandwidth": 0},  # 0
        {"name": "Experiment 1", "bandwidth": 100 * 1024**2},  # 100MB/s
        {"name": "Experiment 2", "bandwidth": 300 * 1024**2},  # 300MB/s
        {"name": "Experiment 3", "bandwidth": 1 * 1024**3},  # 1GB/s
        {"name": "Experiment 4", "bandwidth": 3 * 1024**3},  # 3GB/s
        
    ]
    duration = 10  # Duration for each phase in seconds

    results = []
    for exp in experiments:
        result = run_experiment(
            exp["name"], duration, exp["bandwidth"], core_0=0, core_1=1
        )
        results.append(result)
    power=[]
    print("\nAll Experiments Completed.")
    print("Summary:")
    for idx, res in enumerate(results):
        print(
            f"{experiments[idx]['name']}: Energy Used: {res[0]:.2f} J, Avg Power: {res[1]:.2f} W"
        )
        power.append(res[1])
    import matplotlib.pyplot as plt
    data_speed = [0, 100, 300, 1000, 2000]  # Data speeds in MB/s (0 indicates no communication)
    # power = [15.2, 22.8, 35.6, 50.3, 5.0]  # Power consumption in Watts

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(data_speed, power, marker='o', linestyle='-', linewidth=2, markersize=8, label="Power vs Speed")
    plt.title("Power Consumption vs Data Speed", fontsize=16)
    plt.xlabel("Data Speed (MB/s)", fontsize=14)
    plt.ylabel("Power (Watts)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Show the plot
    plt.savefig("power_vs_speed.png")


if __name__ == "__main__":
    main()
