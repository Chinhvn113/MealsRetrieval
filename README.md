# MealsRetrieval-ROOMELSA
[Technical Report](https://drive.google.com/file/d/1NyK8ZdrA9-NU3vMEIvsJ_BISX6UZwRYo/view?usp=sharing)
## Usage:
We use **Python 3.10.13** and **ubuntu 22.04**

Requires 25+ GB VRAM
### Install Requirements:

```
pip install -r requirements.txt
```
### Dowload dataset
```
gdown --fuzzy 'https://drive.google.com/file/d/10EkNQBO_DBKwsEkNVVeJMS3grHCO1Ruz/view?usp=sharing'
unzip private.zip 
```
### Download inpainting model weights
Download [models](https://huggingface.co/PAIR/Zero-Painter) weights to the `models` folder inside ZeroPainter
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
