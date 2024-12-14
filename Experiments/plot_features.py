import os
import librosa
import matplotlib.pyplot as plt
import numpy
import k_means_constrained


def plot_spectral_flatness(): # Test spectral flatness graph
  (file_data, file_sample_rate) = librosa.load("Datasets/MagnaTagATune/0/american_bach_soloists-j_s__bach__cantatas_volume_v-01-gleichwie_der_regen_und_schnee_vom_himmel_fallt_bwv_18_i_sinfonia-0-29.mp3")
  print(len(file_data))

  # matplotlib test data
  S = librosa.feature.melspectrogram(y=file_data, sr=file_sample_rate, n_mels=128, fmax=8000)

  fig, ax = plt.subplots()
  S_dB = librosa.power_to_db(S, ref=numpy.max)
  img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=file_sample_rate, fmax=8000, ax=ax)
  fig.colorbar(img, ax=ax, format='%+2.0f dB')
  ax.set(title='Mel-frequency spectrogram')
# plot_spectral_flatness()


# Because of dynamic typing, we have to do this.
# Just treat this like a C struct for now.
class MagnaTagATuneSong():
  def __init__(self, name, data, bpm, spectral_flatness, spectral_centroid):
    self.name = name
    self.bpm = bpm
    self.data = data
    self.spectral_flatness = spectral_flatness
    self.spectral_centroid = spectral_centroid

dirs = os.listdir('Datasets/MagnaTagATune')
songs = []
for dir_name in dirs:
  dir = f"Datasets/MagnaTagATune/{dir_name}"
  if not os.path.isdir(dir):
    continue

  song_names = os.listdir(dir)
  for i in range(1): # TODO: Speed this up so we can read in a lot of data at once.

    # Right now, we want reproducability. We'll just use the first song we find.
    # song_name_idx = random.randrange(len(song_names))
    song_name_idx = i

    song_name = song_names[song_name_idx]
    song_path = f"{dir}/{song_name}"
    file_data, file_sample_rate = librosa.load(song_path)
    (bpm_array, _) = librosa.beat.beat_track(y=file_data)

    # Librosa returns an array of bpms, one for each channel of your audio.
    # We should only have a single channel, so assert on that for now.
    assert(isinstance(bpm_array, numpy.ndarray))
    assert(len(bpm_array) == 1)

    # The spectral flatness is a 2D array.
    # It should have number of Fast Fourier Transform "frames".
    # In this example, there is only one frame.
    # TODO: Understand what a Fast Fourier Transform frame is.
    spectral_flatness = librosa.feature.spectral_flatness(y=file_data)
    assert(len(spectral_flatness) == 1)

    # For some reason, librosa.feature.spectral_centroid returns a 2d array,
    # where the first dimension is always zero.
    spectral_centroid = librosa.feature.spectral_centroid(y=file_data)
    assert(len(spectral_centroid) == 1)

    song = MagnaTagATuneSong(
        name=song_path,
        data=file_data,
        bpm=bpm_array[0],
        spectral_flatness=numpy.mean(spectral_flatness[0]),
        spectral_centroid=numpy.mean(spectral_centroid[0]))
    songs.append(song)



def plot_bpm_and_spectral_flatness():

  figure = plt.figure()
  subplot = figure.add_subplot(projection='3d')
  subplot.set_xlabel("bpm")
  subplot.set_ylabel("spectral flatness")
  subplot.set_zlabel("spectral centroid")

  data_py = []
  for song in songs:
    data_py.append([song.bpm, song.spectral_flatness, song.spectral_centroid])

  data = numpy.array(data_py)

  cluster_count = 3

  cluster_size_avg = len(songs) / float(cluster_count)

 # These are values I chose so the clusters aren't too big or
  cluster_size_min = int(cluster_size_avg * 2/3)
  cluster_size_max = int(cluster_size_avg * 1.5)

  clf = k_means_constrained.KMeansConstrained(
    n_clusters=cluster_count,
    size_min=cluster_size_min,
    size_max=cluster_size_max,
    random_state=0
  )

  labels = clf.fit_predict(data)
  clusters = clf.cluster_centers_

  subplot.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap="plasma")
  subplot.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2], marker="X", s=200)
  plt.show()

plot_bpm_and_spectral_flatness()




    
