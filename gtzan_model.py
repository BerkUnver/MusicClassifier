import gtzan
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode


device = None
if torch.cuda.is_available():
  device = torch.device("cuda:0")
  torch.set_default_device(device)
  print("Torch is using CUDA.")
else:
  devie = torch.device("cpu")
  print("Torch is using the CPU.")

start_dir = "Datasets/GTZAN/images_original"
gtzan_file_paths = []
genres = os.listdir(start_dir)
for genre in genres:
  genre_dir = os.path.join(start_dir, genre)
  filenames = os.listdir(genre_dir)
  gtzan_file_paths += [os.path.join(genre_dir, filename) for filename in filenames]


class ClipDataset(Dataset):
  def __init__(self, file_paths, transforms=None):
    self.file_paths = file_paths
    self.transforms = transforms

  def __len__(self):
    return len(self.file_paths)

  def __getitem__(self, ind):
    filename = self.file_paths[ind]
    label = -1
    for ind, genre in enumerate(genres):
      if genre in filename:
        label = ind
    img = Image.open(filename).convert('L')
    img = transforms.functional.crop(img, top=40, left=55, height=216, width=336)
    if self.transforms:
      img = self.transforms(img)
    return img, label



train_transforms = transforms.Compose([
transforms.ToTensor(),
transforms.Resize((216, 336))
])



gtzan_training = ClipDataset(gtzan_file_paths, transforms=train_transforms)
img, label = gtzan_training.__getitem__(700)


img = Image.open('Datasets/GTZAN/images_original/blues/blues00001.png')


train_loader = DataLoader(gtzan_training, batch_size=32, shuffle=True, generator=torch.Generator(device=device))


class MusicClassifierGTZAN(nn.Module):
  def __init__(self, num_clusters=10, latent_dim=128):
    super(MusicClassifierGTZAN, self).__init__()
    self.encoder = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512*14*21, latent_dim)
    )

    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, 512*14*21),
        nn.ReLU(),
        nn.Unflatten(1, (512, 14, 21)),
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1,output_padding=(0, 1)),
        nn.ReLU(),
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU()
    )

    self.cluster_centers = nn.Parameter(torch.randn(num_clusters, latent_dim))

  def encode(self, x):
    return self.encoder(x)

  def decode(self, x):
    return self.decoder(x)

  def forward(self, x):
    latent = self.encode(x)
    out = self.decode(latent)
    return latent, out

  def soft_assignments(self, latent):
    q = 1.0 / (1.0 + torch.sum((latent.unsqueeze(1) - self.cluster_centers)**2, dim=2))
    q /= q.sum(dim=1, keepdim=True)
    return q


def clustering_loss(z, cluster_centers, q):
  p = (q**2) / q.sum(dim=0)
  p /= p.sum(dim=1, keepdim=True)
  return F.kl_div(q.log(), p, reduction='batchmean');

 
def train_model(model, device, trainloader, optimizer, num_epochs):

  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
      inputs = inputs.to(device)
      latent, outputs = model(inputs)

      # calculating composite loss
      recon_loss = F.mse_loss(outputs, inputs)
      q = model.soft_assignments(latent)
      cluster_loss = clustering_loss(latent, model.cluster_centers, q)

      loss = recon_loss + (0.1 * cluster_loss)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      print(f"{i+1}th batch successfully passed")
    print(f"Epoch {epoch+1}, Loss: {(running_loss/len(trainloader)): .4f}")

model = MusicClassifierGTZAN(num_clusters=10, latent_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_model(model, device, train_loader, optimizer, 5)


gtzan_training, gtzan_eval = train_test_split(gtzan_training, test_size=0.2, random_state=42)
print(len(gtzan_training), len(gtzan_eval))


eval_loader = DataLoader(gtzan_eval, batch_size=32, shuffle=True, generator=torch.Generator(device=device))


def map_clusters_to_labels(clusters, labels):
  mapping = np.zeros_like(clusters)
  for cluster in np.unique(clusters):
    cluster_inds = np.where(clusters == cluster)[0]
    labels_in_cluster = labels[cluster_inds]
    most_freq = mode(labels_in_cluster)[0]
    mapping[cluster_inds] = most_freq
  return mapping


kmeans = KMeans(n_clusters=10, random_state=42)
for i, (inputs, labels) in enumerate(eval_loader):
  if i != len(eval_loader)-1:
    inputs = inputs.to(device)
    latent = model.encode(inputs)
    label = labels
    latent = latent.cpu()
    kmeans.fit(latent.detach().numpy())
    clusters = kmeans.labels_
    mapped_clusters = map_clusters_to_labels(clusters, label.cpu())
    print("labels:", label)
    print("clusters:", mapped_clusters)
    print(accuracy_score(label.cpu(), mapped_clusters))
