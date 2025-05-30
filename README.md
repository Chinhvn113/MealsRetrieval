# MealsRetrieval-ROOMELSA
[Technical Report](https://drive.google.com/file/d/1NyK8ZdrA9-NU3vMEIvsJ_BISX6UZwRYo/view?usp=sharing)
## Usage:
We use **Python 3.10.13**, **Ubuntu 22.04**, **cuda version 12.8** and **torch2.6.0**

Requires 27+ GB VRAM
### Install Requirements:
First install pytorch
`pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126`

Then go to this [link](https://github.com/Dao-AILab/flash-attention/releases) to get the suitable wheel for flash-attn library, if theres no suiltable wheel, use `pip install flash-attn --no-build-isolation` or proceed without it
(Missing flash-attn may downgrades our models performance).

Script for torch2.6 and cuda12x `pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3%2Bcu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"`

After that, install all requirements:
```
pip install -r requirements.txt
```
### Download dataset
```
gdown --fuzzy 'https://drive.google.com/file/d/10EkNQBO_DBKwsEkNVVeJMS3grHCO1Ruz/view?usp=sharing'
unzip private.zip 
```
### Download inpainting model weights
Download [models](https://huggingface.co/PAIR/Zero-Painter) weights to the `models` folder inside ZeroPainter
```
sudo apt update
sudo apt install git-lfs
git lfs install
```
```
cd ZeroPainter
mkdir models
git clone https://huggingface.co/PAIR/Zero-Painter
mv Zero-Painter/* models
cd ..
```
## Data Preprocessing
### Download blender:
- Windows: Download using this link
 https://mirror.freedif.org/blender/release/Blender4.4/blender-4.4.0-windows-x64.zip
- Linux: Enter this command in terminal
```
wget https://mirror.freedif.org/blender/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz
```
**Download and extract the Blender folder into the Data_preprocessing folder.**

### Start Preprocessing:
**Objects folder format should look like this :**
```
/path/to/objects/
│   └── obj1/  
│   └── obj2/ 
```
#### If you are using Windows, use Git Bash instead of CMD/PowerShell.
Grant permissions: 
```
chmod +x Data_preprocessing/run.sh
```
Run:
Linux:
```
Data_preprocessing/run_linux.sh /path/to/your/objects/folder/
```
Windows:
```
Data_preprocessing/run_win.sh /path/to/your/objects/folder/
```

After rendering:
```
/path/to/objects/
│   └── obj1/
|           rendered_images/
|           caption.txt
|           .....
│   └── obj2/
|           rendered_images/
|           caption.txt
|           .....
```
## Construct database:
```
python Database/faiss_retrieve_demo.py --object_dir path/to/objects/ --index_save_dir path/to/indexfolder #save index database into a folder
```
## Run inference:
```
python main.py --input path/to/privatedataset --index_save_dir path/to/indexfolder
```
