# Install NVIDIA drivers (G5 instances use NVIDIA A10G GPUs)
sudo dnf install kernel-modules-extra.x86_64

# Clean up any existing NVIDIA installations
sudo dnf remove -y "*nvidia*" "*cuda*"

# Install required dependencies
sudo dnf install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r) gcc make dkms

# Add the NVIDIA CUDA repository
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

# Clean package cache
sudo dnf clean all

# Install NVIDIA driver (latest version with DKMS for kernel updates)
sudo dnf module install -y nvidia-driver:latest-dkms

# Install CUDA toolkit
sudo dnf install -y cuda

# Blacklist nouveau driver to prevent conflicts
echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf

# Update initramfs
sudo dracut --force

# Set up environment variables for CUDA
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reboot the system to load the new driver
sudo reboot


# Set up environment variables for CUDA
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc