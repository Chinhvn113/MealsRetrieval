# MealsRetrieval-ROOMELSA

## Usage:
### Install Requirements:
```
pip install -r requirements.txt
```
### Dowload dataset
```
gdown --fuzzy 'https://drive.google.com/file/d/10EkNQBO_DBKwsEkNVVeJMS3grHCO1Ruz/view?usp=sharing'
unzip private.zip 
```
## Data Preprocessing
### download blender:
- windown: Download using this link
 https://mirror.freedif.org/blender/release/Blender4.4/blender-4.4.0-windows-x64.zip
- linux: enter this command in terminal
```
wget https://mirror.freedif.org/blender/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz
```
**Download and extract the Blender folder into the Data_preprocessing folder.**

### Start Preprocessing:
#### If you are using Windows, use Git Bash instead of CMD/PowerShell.
Grant permissions: 
```
chmod +x Data_preprocessing/run.sh
```
Run:
```
Data_preprocessing/run.sh /path/to/your/objects/folder/
```
**Objects folder format should look like this :**
```
/path/to/objects/
│   └── obj1/  
│   └── obj2/ 
```
after rendering:
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
