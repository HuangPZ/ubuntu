import os
import time
import threading
import queue

import pynvml
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import subprocess
import matplotlib.pyplot as plt



# def power_monitor(handle, stop_event, data_queue, interval=0.05):
#     """
#     Continuously sample the instantaneous power draw (mW) from NVML
#     and store (timestamp, power_mW) in a queue until stop_event is set.
#     """
#     while not stop_event.is_set():
#         power_mW = pynvml.nvmlDeviceGetPowerUsage(handle)  # instantaneous power (mW)
#         timestamp = time.time()
#         data_queue.put((timestamp, power_mW))
#         # time.sleep(interval)
def power_monitor(device_id, stop_event, data_queue, interval=0.05):
    # """
    # Continuously sample the instantaneous power draw (mW) using `nvidia-smi`
    # and store (timestamp, power_mW) in a queue until stop_event is set.

    # Parameters:
    #     device_id (int): The GPU ID to monitor.
    #     stop_event (threading.Event): Event to signal stopping the monitor.
    #     data_queue (queue.Queue): Queue to store (timestamp, power_mW) samples.
    #     interval (float): Sampling interval in seconds (default: 0.05s).
    # """
    # device_id = 0
    # while not stop_event.is_set():
    #     try:
    #         # Run `nvidia-smi` command to get power draw for the specific GPU
    #         result = subprocess.run(
    #             ["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits"],#csv","--loop-ms=80"],
    #             stdout=subprocess.PIPE,
    #             stderr=subprocess.PIPE,
    #             text=True
    #         )

    #         # Parse the output (power in watts)
    #         power_w = float(result.stdout.strip())
    #         power_mw = power_w #* 1000  # Convert watts to milliwatts

    #         # Get the current timestamp
    #         timestamp = time.time()

    #         # Add the sample to the data queue
    #         data_queue.put((timestamp, power_mw))

    #     except Exception as e:
    #         print(f"Error during power monitoring: {e}")
    #         break

    #     # Wait for the next interval
    #     # time.sleep(interval)
    try:
        # Start nvidia-smi in continuous mode with a 10 ms interval
        process = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "--loop-ms=10"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        while not stop_event.is_set():
            line = process.stdout.readline().strip()
            if line:
                try:
                    power_w = float(line)
                    timestamp = time.time()
                    data_queue.put((timestamp, power_w * 1000))  # Convert watts to milliwatts
                except ValueError:
                    print(f"Invalid output: {line}")

        process.terminate()
    except Exception as e:
        print(f"Error during power monitoring: {e}")


def conv2d_im2col_mpc(a1, W1, a2, W2, conv_module):
    """
    Mimic a Conv2d forward pass using 'im2col' + matrix multiplication:
       c = (a1_im2col @ W2_reshaped) + (a2_im2col @ W1_reshaped).
    """
    # Unfold (im2col) the input
    # shape: [N, inC*kH*kW, outH*outW]
    a1_unf = F.unfold(
        a1,
        kernel_size=conv_module.kernel_size,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        stride=conv_module.stride
    )
    a2_unf = F.unfold(
        a2,
        kernel_size=conv_module.kernel_size,
        dilation=conv_module.dilation,
        padding=conv_module.padding,
        stride=conv_module.stride
    )

    # Reshape the kernel for matmul
    # original shape: [outC, inC, kH, kW]
    # reshape to:     [outC, inC*kH*kW]
    outC, inC, kH, kW = W1.shape
    W1_mat = W1.view(outC, inC * kH * kW)
    W2_mat = W2.view(outC, inC * kH * kW)

    # Compute unfolded output (matrix multiplication)
    # a1_unf has shape [N, inC*kH*kW, outH*outW]
    # W1_mat has shape [outC, inC*kH*kW]
    # Result: [N, outC, outH*outW]
    c_unf = torch.bmm(W1_mat.unsqueeze(0).expand(a1_unf.size(0), -1, -1), a2_unf) + \
            torch.bmm(W2_mat.unsqueeze(0).expand(a1_unf.size(0), -1, -1), a1_unf)

    # Reshape unfolded output for folding
    c_unf = c_unf.permute(0, 2, 1).contiguous()

    # Calculate output shape from convolution parameters
    outH = ((a1.size(2) + 2 * conv_module.padding[0] - conv_module.dilation[0] * (conv_module.kernel_size[0] - 1) - 1) // conv_module.stride[0]) + 1
    outW = ((a1.size(3) + 2 * conv_module.padding[1] - conv_module.dilation[1] * (conv_module.kernel_size[1] - 1) - 1) // conv_module.stride[1]) + 1
    # print(outH, outW, c_unf.shape)
    # # Fold back to original spatial shape
    # c = F.fold(
    #     c_unf.transpose(1, 2),
    #     output_size=(outH, outW),
    #     kernel_size=conv_module.kernel_size,
    #     dilation=conv_module.dilation,
    #     padding=conv_module.padding,
    #     stride=conv_module.stride
    # )

    return c_unf #c


def measure_energy_resnet50_conv_linear_mpc(
    gpu_index=0,
    batch_size=4,
    num_iterations=1,
    sampling_interval=0.05
):
    """
    Fixed version:
    1. Generate all data first (inputs and weights for all layers).
    2. Start power monitoring after data generation.
    3. Perform computations.
    """
    #---------------------------------------------------------------------
    # 1) Restrict ourselves to a specific GPU, initialize device and NVML
    #---------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    device = torch.device("cuda:0")  # "cuda:0" now refers to that selected GPU

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0 => the selected GPU

    # #---------------------------------------------------------------------
    # # 2) Load ResNet-50, find conv & linear layers
    # #---------------------------------------------------------------------
    # model = models.resnet50(pretrained=False).to(device)
    # model.eval()

    # layers = []
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         layers.append(("conv", module))
    #     elif isinstance(module, nn.Linear):
    #         layers.append(("linear", module))

    # print("Discovered these convolution/linear layers in ResNet-50:")
    # for (lyr_type, lyr_module) in layers:
    #     print(f" - {lyr_type}: {lyr_module}")

    # if not layers:
    #     print("No conv or linear layers found (unexpected). Exiting.")
    #     return

    # #---------------------------------------------------------------------
    # # 3) Generate all data first
    # #---------------------------------------------------------------------
    # data_store = []  # To store generated data for all layers

    # with torch.no_grad():
    #     for idx, (lyr_type, lyr_mod) in enumerate(layers):
    #         if lyr_type == "linear":
    #             # Generate input activations and weights for linear layer
    #             in_features = lyr_mod.in_features
    #             out_features = lyr_mod.out_features

    #             a1_gpu = torch.randn(batch_size, in_features, device=device)
    #             W1_gpu = torch.randn(in_features, out_features, device=device)

    #             a2_cpu = torch.randn(batch_size, in_features)
    #             W2_cpu = torch.randn(in_features, out_features)

    #             data_store.append((lyr_type, a1_gpu, W1_gpu, a2_cpu, W2_cpu))

    #         elif lyr_type == "conv":
    #             # Generate input activations and weights for conv layer
    #             conv_mod = lyr_mod
    #             inC = conv_mod.in_channels
    #             outC = conv_mod.out_channels

    #             # Assuming fixed spatial size (e.g., 112x112)
    #             H, W = 112, 112

    #             a1_gpu = torch.randn(batch_size, inC, H, W, device=device)
    #             W1_gpu = torch.randn(outC, inC, *conv_mod.kernel_size, device=device)

    #             a2_cpu = torch.randn(batch_size, inC, H, W)
    #             W2_cpu = torch.randn(outC, inC, *conv_mod.kernel_size)

    #             data_store.append((lyr_type, a1_gpu, W1_gpu, a2_cpu, W2_cpu, conv_mod))

    # print("\nAll data generated and stored. Starting computations...\n")

    #---------------------------------------------------------------------
    # 4) Start background thread to sample power usage
    #---------------------------------------------------------------------
    stop_event = threading.Event()
    data_queue = queue.Queue()
    monitor_thread = threading.Thread(
        target=power_monitor,
        args=(handle, stop_event, data_queue, sampling_interval)
    )

    monitor_thread.start()
    overall_start_time = time.time()

    #---------------------------------------------------------------------
    # 5) Perform computations for all layers
    #---------------------------------------------------------------------
    with torch.no_grad():
        for it in range(num_iterations):
            
            print(f"\n---- Iteration {it+1}/{num_iterations} ----")
            time.sleep(10)  # Sleep for 1s between iterations
            continue
            for idx, data in enumerate(data_store):
                if data[0] == "linear":
                    # Retrieve stored data
                    _, a1_gpu, W1_gpu, a2_cpu, W2_cpu = data

                    # Move share2 to GPU
                    a2_gpu = a2_cpu.to(device, non_blocking=True)
                    W2_gpu = W2_cpu.to(device, non_blocking=True)

                    # c = a1*W2 + a2*W1
                    c_gpu = a1_gpu.mm(W2_gpu) + a2_gpu.mm(W1_gpu)

                    # Move c_gpu back to CPU
                    c_cpu = c_gpu.to("cpu", non_blocking=True)
                    torch.cuda.synchronize()

                    # print(f"  [Linear #{idx+1}] done.")

                elif data[0] == "conv":
                    # Retrieve stored data
                    _, a1_gpu, W1_gpu, a2_cpu, W2_cpu, conv_mod = data

                    # Move share2 to GPU
                    a2_gpu = a2_cpu.to(device, non_blocking=True)
                    W2_gpu = W2_cpu.to(device, non_blocking=True)

                    # c = a1*W2 + a2*W1, but via im2col approach
                    c_gpu = conv2d_im2col_mpc(a1_gpu, W1_gpu, a2_gpu, W2_gpu, conv_mod)

                    # Move c_gpu back to CPU
                    c_cpu = c_gpu.to("cpu", non_blocking=True)
                    torch.cuda.synchronize()

                    out_shape = c_gpu.shape
                    # print(f"  [Conv #{idx+1}] output={list(out_shape)} done.")

    overall_end_time = time.time()

    #---------------------------------------------------------------------
    # 6) Stop monitoring, collect samples
    #---------------------------------------------------------------------
    stop_event.set()
    monitor_thread.join()

    samples = []
    while not data_queue.empty():
        samples.append(data_queue.get())
    samples.sort(key=lambda s: s[0])

    #---------------------------------------------------------------------
    # 7) Integrate power over time to estimate total energy
    #---------------------------------------------------------------------
    time_points = []
    power_points_mW = []

    total_energy_J = 0.0
    for (ts, power_mW) in samples:
        time_points.append(ts)
        power_points_mW.append(power_mW)

    t_all = 0
    for i in range(len(samples) - 1):
        t0, p0_mW = samples[i]
        t1, p1_mW = samples[i + 1]
        dt = t1 - t0  # seconds
        p_avg_W = 0.5 * (p0_mW + p1_mW) * 1e-3  # Average power (W)
        total_energy_J += (p_avg_W * dt)
        t_all += dt

    duration_s = overall_end_time - overall_start_time
    

    print("\n==============================================")
    print(f"Total run time: {duration_s:.4f} s, or {t_all:.4f} s (power samples)")
    print(f"Approx. total energy: {total_energy_J:.4f} J")
    print("==============================================")

    #---------------------------------------------------------------------
    # 8) Plot power (mW) over time
    #---------------------------------------------------------------------
    if len(time_points) > 1:
        t0 = time_points[0]
        shifted_times = [t - t0 for t in time_points]
        print(len(shifted_times), len(power_points_mW))
        # print(shifted_times)
        plt.figure(figsize=(8, 5))
        plt.plot(shifted_times, power_points_mW, label="Power (W)")
        plt.xlabel("Time (s) since start")
        plt.ylabel("Power draw (W)")
        plt.title("GPU Power Usage During MPC-like Ops (Conv + Linear) in ResNet-50")
        plt.legend()
        plt.tight_layout()
        plt.savefig("power_usage.png")
    else:
        print("Not enough samples to plot power usage.")

    # Shutdown NVML
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    measure_energy_resnet50_conv_linear_mpc(
        gpu_index=0,        # Use GPU 0
        batch_size=4,       # Example batch size
        num_iterations=1,   # Perform a couple of iterations
        sampling_interval=0.05
    )
