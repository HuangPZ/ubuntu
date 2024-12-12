import os
import time
import subprocess

def read_rapl_energy(cpu=0):
    """Reads the RAPL energy register using msr-tools."""
    try:
        register = "0x611"  # RAPL energy register for the package domain
        result = subprocess.run(
            ["sudo", "rdmsr", "-p", str(cpu), register],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            print(f"Error reading RAPL energy: {result.stderr.decode().strip()}")
            return None
        return int(result.stdout.strip(), 16)  # Convert hex to int
    except Exception as e:
        print(f"Error: {e}")
        return None

def cpu_intensive_task():
    """Performs a CPU-intensive task."""
    print("Running CPU-intensive task...")
    total = 0
    for i in range(1, 10**7):
        total += i ** 0.5
    print("Task complete.")

def measure_energy_usage():
    """Measures energy usage during a task."""
    print("Measuring RAPL energy usage...")
    start_energy = read_rapl_energy()
    if start_energy is None:
        print("RAPL energy not available.")
        return

    start_time = time.time()
    cpu_intensive_task()
    end_time = time.time()

    end_energy = read_rapl_energy()
    if end_energy is None:
        print("RAPL energy not available.")
        return

    energy_used = (end_energy - start_energy) & 0xFFFFFFFF  # Handle register overflow
    print(f"Energy used: {energy_used} units (approx.)")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    measure_energy_usage()
