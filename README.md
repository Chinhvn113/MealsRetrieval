
# Usage:
### Install Requirements:
```
pip install -r requirements.txt
```
### Dowload dataset
```
gdown 'https://drive.google.com/file/d/10EkNQBO_DBKwsEkNVVeJMS3grHCO1Ruz/view?usp=sharing'
unzip private.zip 
```
### download blender:
- windown: Download using this link
 https://mirror.freedif.org/blender/release/Blender4.4/blender-4.4.0-windows-x64.zip
- linux: enter this command in terminal
```
wget https://mirror.freedif.org/blender/release/Blender4.4/blender-4.4.0-linux-x64.tar.xz
```
**Download and extract the Blender folder into the Data_preprocessing folder.**

### Start Preprocessing:
Grant permissions: 
```
chmod +x run.sh
```
Run:
```
./run.sh /path/to/your/objects/forder/
```
**Objects folder format should look like this :**
```
/path/to/objects/
│   └── obj1/  
│   └── obj2/ 
```
###Construct database:
```
python Database/faiss_retrive_demo.py --object_dir path/to/your/object --index_save_dir path/to/your/indexfolder #save index database into a folder
```
### Run inference:
```
python main.py --input path/to/privatedataset --index_save_dir path/to/your/indexfolder
```
