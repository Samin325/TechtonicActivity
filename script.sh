#!/bin/bash

python3 -m venv env_test
source env_test/bin/activate
pip3 install torch torchvision torchaudio
pip install obspy pandas scikit-learn
pip install gdown
gdown https://drive.google.com/file/d/19lfv8udhWZcBmoHpISQfvlD1FOVBp5mg/view?usp=drive_link
unzip space_apps_2024_seismic_detection.zip 
rm -rf space_apps_2024_seismic_detection.zip 
cd /usr/local/cuda-12.1/lib64
sudo rm -f libcudnn*
cd /usr/local/cuda-12.1/include
sudo rm -f cudnn*
cd ~/TechtonicActivity/
