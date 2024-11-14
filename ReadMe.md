Download the following folder from Google Drive:

https://drive.google.com/drive/folders/1fEJeKYphScj5x9L808BDD6F-ffSY8PKf?usp=drive_link

Google drive will download this as two separate zip files because they are so big.
Unzip them both into the project directory, and it should work.

# Setup
To get started (on Windows), create a virtual environment in your repository folder using: 
```
python -m venv env
```
In your terminal (cmd), you can activate the virtual environment using:
```
env/Scripts/activate
```
On Powershell, use activate.ps1 instead. To close your environment, use 
```
deactivate
```
Finally, while your virtual environment is active, make sure you have all the dependencies installed with
```
pip install -r requirements.txt
```
Sidenote: there is currently an issue with installing numpy due to different dependencies requiring different versions. When numba 0.61 comes out the issue should be resolved.
