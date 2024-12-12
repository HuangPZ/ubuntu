import threading
import time
import psutil
import os
import subprocess
from cryptography.fernet import Fernet


def bind_to_cpu(core_id):
    """Bind the current process to a specific CPU core."""
    try:
        os.sched_setaffinity(0, {core_id})
    except AttributeError:
        print(f"Unable to bind to CPU core {core_id}. Ignoring.")


def simulate_communication(phase_name, duration, bandwidth, core_id, encrypt=False):
    """
    Simulate communication workload (upload/download) with optional encryption.
    Args:
        phase_name (str): Name of the phase (e.g., "Upload 100MB/s").
        duration (int): Duration of the experiment in seconds.
        bandwidth (int): Bandwidth in bytes per second.
        core_id (int): CPU core to bind the process.
        encrypt (bool): Whether to simulate encryption/decryption.
    """
    print(f"{phase_name} started on CPU core {core_id}.")

    # Bind to the specified core
    bind_to_cpu(core_id)

    # Simulate encryption/decryption setup
    key = Fernet.generate_key()
    cipher = Fernet(key)

    # Simulate data transmission
    data_transmitted = 0
    chunk_size = 10 * 1024 * 1024  # 10 MB
    start_time = time.time()

    while time.time() - start_time < duration:
        data_chunk = os.urandom(chunk_size)  # Generate random data
        if encrypt:
            encrypted_chunk = cipher.encrypt(data_chunk)  # Encrypt data
            decrypted_chunk = cipher.decrypt(encrypted_chunk)  # Decrypt data
        data_transmitted += chunk_size

    # Ensure total data doesn't exceed bandwidth * duration
    data_transmitted = min(data_transmitted, bandwidth * duration)

    print(f"{phase_name} completed. Data transmitted: {data_transmitted / 1e6:.2f} MB")
    return data_transmitted

def measure_energy():
    """Measures energy using RAPL with perf."""
    try:
        # Measure energy over a short duration for better precision
        result = subprocess.run(
            ["sudo", "perf", "stat", "-e", "power/energy-pkg/", "sleep", "1"],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        
        output = result.stderr.decode("utf-8")
        print(output)
        for line in output.splitlines():
            if "power/energy-pkg/" in line:
                # Extract energy value in Joules
                return float(line.split()[0])
    except Exception as e:
        print(f"Failed to measure energy: {e}")
    return None


def measure_cpu_usage(duration, core_id, result_list, index):
    """Measures average CPU usage for a specific core during the phase."""
    print(f"Measuring CPU usage on core {core_id}...")
    usage = []
    start_time = time.time()
    while time.time() - start_time < duration:
        usage.append(psutil.cpu_percent(interval=1, percpu=True)[core_id])
    result_list[index] = sum(usage) / len(usage)


def run_experiment(name, duration, bandwidth, core_0, core_1):
    """
    Run a single experiment with two threads:
    - One thread for upload (on core_0).
    - One thread for download (on core_1).
    """
    print(f"\nStarting experiment: {name}")

    # Record energy before the experiment
    energy_before = measure_energy()
    if energy_before is None:
        print(f"Failed to record initial energy for {name}")
        return {"name": name, "cpu_0_usage": None, "cpu_1_usage": None, "energy_used": None}

    # Results storage for CPU usage
    cpu_usage_results = [0, 0]

    # Threads for CPU usage and communication
    cpu_thread_0 = threading.Thread(
        target=measure_cpu_usage, args=(duration, core_0, cpu_usage_results, 0)
    )
    cpu_thread_1 = threading.Thread(
        target=measure_cpu_usage, args=(duration, core_1, cpu_usage_results, 1)
    )
    upload_thread = threading.Thread(
        target=simulate_communication,
        args=(f"Upload {bandwidth / 1e6:.2f} MB/s", duration, bandwidth, core_0, True),
    )
    download_thread = threading.Thread(
        target=simulate_communication,
        args=(f"Download {bandwidth / 1e6:.2f} MB/s", duration, bandwidth, core_1, True),
    )

    # Start threads
    cpu_thread_0.start()
    cpu_thread_1.start()
    upload_thread.start()
    download_thread.start()

    # Wait for threads to complete
    cpu_thread_0.join()
    cpu_thread_1.join()
    upload_thread.join()
    download_thread.join()

    # Record energy after the experiment
    energy_after = measure_energy()
    if energy_after is None:
        print(f"Failed to record final energy for {name}")
        energy_used = None
    else:
        energy_used = max(0, energy_after - energy_before)  # Ensure no negative energy

    print(f"\nExperiment {name} results:")
    print(f"CPU 0 Usage: {cpu_usage_results[0]:.2f}%")
    print(f"CPU 1 Usage: {cpu_usage_results[1]:.2f}%")
    print(f"Energy Used: {energy_used} Joules\n")

    return {
        "name": name,
        "cpu_0_usage": cpu_usage_results[0],
        "cpu_1_usage": cpu_usage_results[1],
        "energy_used": energy_used,
    }


def main():
    # Experiment configurations
    experiments = [
        {"name": "Experiment 1", "bandwidth": 100 * 1024**2},  # 100MB/s
        {"name": "Experiment 2", "bandwidth": 1 * 1024**3},  # 1GB/s
        {"name": "Experiment 3", "bandwidth": 10 * 1024**3},  # 10GB/s
        {"name": "Experiment 4", "bandwidth": 0},  # 1GB/s
    ]
    duration = 10  # Duration for each phase in seconds

    results = []
    for exp in experiments:
        result = run_experiment(
            exp["name"], duration, exp["bandwidth"], core_0=0, core_1=1
        )
        results.append(result)

    print("\nAll Experiments Completed.")
    print("Summary:")
    for res in results:
        print(
            f"{res['name']}: CPU 0 Usage: {res['cpu_0_usage']:.2f}%, "
            f"CPU 1 Usage: {res['cpu_1_usage']:.2f}%, Energy Used: {res['energy_used']} Joules"
        )


if __name__ == "__main__":
    main()
