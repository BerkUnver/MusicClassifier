Download the following folder from Google Drive:

https://drive.google.com/drive/folders/1fEJeKYphScj5x9L808BDD6F-ffSY8PKf?usp=drive_link

Unzip it into this project directory. There should be a directory named Datasets with GTZAN inside.

# Runnable scripts
Navigate to the main directory of this project.

Run the clustering based on bpm and spectral centroid (Berk Unver's contribution)
```bash
python gtzan_bpm_and_specular_centroid.py
```

Generate the tables ranking each genre by feature (Berk Unver's contribution)
```bash
python gtzan_genres_ranked.py
```

Run Principle Component Analysis and K-Means (Berk Unver's contribution)
```bash
python gtzan_pca.py
```

Run the neural network (Sunny Chan's contribution)
```bash
python gtzan_model.py
```

# Setup
To get started (on Windows), create a virtual environment in your repository folder using: 
```
python -m venv env
```
In your terminal (cmd), you can activate the virtual environment using:
```
env\Scripts\activate
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
