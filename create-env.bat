@echo off

REM Create the virtual environment
C:\Users\btripp\miniconda3\python.exe -m venv env

REM Activate the virtual environment
call env\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

pip install numpy==2.1.0 wheel setuptools
C:\Users\bento\miniconda3\_conda.exe install gdal

REM Install the required packages
pip install ipykernel jupyter nbformat==5.9.1 requests python-dotenv scipy pandas geopandas osmnx ^
shapely fiona pyproj rasterio rasterstats pysal rioxarray cartopy folium leafmap owslib earthpy ^
sentinelsat planetary_computer pystac_client python-box

REM Install PyTorch (CPU version) and torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

REM Install additional dependencies for Clay
pip install einops==0.7.0 vit-pytorch==1.7.12 install torchdata==0.7.1 torchgeo
pip install zarr==2.16.1 pyyaml matplotlib scikit-learn lightning h5netcdf ^
jsonargparse lancedb pyarrow s3fs scikit-image stackstac timm  transformers ^
typeshed-client wandb

REM Install package to connect to GRASS
pip install grass-session

REM git clone https://github.com/Clay-foundation/model.git

pip freeze > requirements.txt

echo "Virtual environment setup is complete."

REM call with: create-env.bat