#!/bin/bash

# Update and install required packages
sudo apt update
sudo apt install -y linux-tools-common linux-tools-generic net-tools msr-tools
sudo apt install -y $(uname -r | awk '{print "linux-modules-"$1" linux-modules-extra-"$1}')

# Load required kernel modules
sudo modprobe intel_rapl_common
sudo modprobe msr

# Adjust system settings
sudo sysctl -w net.ipv4.tcp_slow_start_after_idle=0

# Verify powercap directory
ls /sys/class/powercap || echo "Powercap directory not found!"

# Install Miniconda
MINICONDA_DIR=~/miniconda3
mkdir -p $MINICONDA_DIR
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $MINICONDA_DIR/miniconda.sh
bash $MINICONDA_DIR/miniconda.sh -b -u -p $MINICONDA_DIR
rm $MINICONDA_DIR/miniconda.sh
source $MINICONDA_DIR/bin/activate

# Create and activate conda environment
conda env create -f environment.yml
conda activate rapl-test

# Install Python packages
pip install psutil pyRAPL pymongo pandas ThrottledSocket numba
conda install -c conda-forge cryptography -y
conda install scapy -y

echo "Setup complete!"

# git config --global user.name "HuangPZ"
# git config --global user.email "670413709@qq.com"