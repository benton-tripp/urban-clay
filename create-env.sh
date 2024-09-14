#!/bin/bash

# Update the system
sudo apt-get update

# Install system dependencies (Python, GDAL, and development tools)
sudo apt-get install -y python3 python3-venv python3-pip gdal-bin libgdal-dev \
    build-essential curl wget unzip

# Install the NVIDIA driver for GeForce RTX 2080 Ti
sudo apt-get install -y nvidia-driver-525

# Add NVIDIA package repositories (for CUDA and cuDNN)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# Add CUDA repository
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt-get update

# Install CUDA toolkit (this will install CUDA 11.8)
sudo apt-get install -y cuda

# Set environment variables for CUDA
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Verify CUDA installation
nvcc --version

# Create a virtual environment
python3 -m venv ~/env

# Activate the virtual environment
source ~/env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install numpy, wheel, setuptools
pip install numpy==2.1.0 wheel setuptools

# Install additional required packages
pip install ipykernel jupyter nbformat==5.9.1 requests python-dotenv scipy pandas geopandas osmnx \
    shapely fiona pyproj rasterio rasterstats pysal rioxarray cartopy folium leafmap owslib earthpy \
    sentinelsat planetary_computer pystac_client

# Install PyTorch (GPU version with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies for Clay
pip install einops==0.7.0 vit-pytorch==1.7.12 torchdata==0.7.1 torchgeo \
    zarr==2.16.1 pyyaml matplotlib scikit-learn lightning h5netcdf jsonargparse \
    lancedb pyarrow s3fs scikit-image stackstac timm transformers typeshed-client wandb

# Optionally, add virtual environment activation to ~/.bashrc
if ! grep -q "source ~/env/bin/activate" ~/.bashrc; then
    echo "source ~/env/bin/activate" >> ~/.bashrc
fi

# Inform the user that the setup is complete
echo "Setup is complete. To activate the virtual environment in future sessions, use 'source ~/env/bin/activate'."
echo "You can check if PyTorch detects the GPU by running 'python -c \"import torch; print(torch.cuda.is_available())\"' inside the virtual environment."
