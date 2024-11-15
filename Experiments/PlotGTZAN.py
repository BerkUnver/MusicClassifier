import os
import numpy
import pandas
import matplotlib.pyplot as plt
import k_means_constrained

# This is just a basic example to show how we can do k-means clustering 
# on a per-feature basis. A future step we can take is to figure out
# which of the gtzan pre-extracted features most closely correspond to 
# genre, which may have some interesting results. I'm not sure if that 
# will involve deep learning or if we can do that with more 
# conventional algorithms.
#           -berk, November 15, 2024

def plot_gtzan():
    frame = pandas.read_csv("Datasets/GTZAN/features_30_sec.csv")

    try:
        bpm = frame["tempo"]
        spectral_centroid = frame["spectral_centroid_mean"]
    except ValueError:
        assert False

    # No spectral flatness in the GTZAN pre-extracted features so we will avoid that for now.
    

    gtzan_genre_count = 10
    gtzan_songs_per_genre = 100

    clf = k_means_constrained.KMeansConstrained(
        n_clusters = gtzan_genre_count,
        size_min = int(gtzan_songs_per_genre * 2/3),
        size_max = int(gtzan_songs_per_genre * 1.5),
        random_state = 0
    )
    
    data = numpy.stack((bpm, spectral_centroid), axis=-1)
    cluster_labels = clf.fit_predict(data)
    
    _, ((cluster_plot, normalized_plot), (genre_plot, _)) = plt.subplots(ncols=2, nrows=2)
    
    point_size = 10
    label_size = 100

    cluster_plot.set_title("Cluster plot")
    cluster_plot.set_xlabel("bpm")
    cluster_plot.set_ylabel("spectral centroid")
    cluster_plot.scatter(bpm, spectral_centroid, c=cluster_labels, cmap="plasma", s=point_size)
    cluster_plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=label_size)

    
    # bpm and spectral centroids are in drastically different units.
    # This results in the clustering being very linear, as seen in
    # the non-normalized graph. To fix this, we will normalize the 
    # bpm and spectral centroid to be between 0 and 1, and then cluster.
    #           -berk, November 11, 2024

    def numpy_normalize(array):
        return (array - numpy.min(array)) / (numpy.max(array) - numpy.min(array))

    normalized_bpm = numpy_normalize(bpm)
    normalized_spectral_centroid = numpy_normalize(spectral_centroid)
    normalized_data = numpy.stack((normalized_bpm, normalized_spectral_centroid), axis=-1)
    normalized_labels = clf.fit_predict(normalized_data)

    normalized_plot.set_title("Normalized cluster plot")
    normalized_plot.set_xlabel("normalized bpm")
    normalized_plot.set_ylabel("normalized spectral centroid")

    normalized_plot.scatter(normalized_bpm, normalized_spectral_centroid, c=normalized_labels, cmap="plasma", s=point_size)
    normalized_plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=label_size)

    
    # The songs are in order, with the first 100 songs being one genre,
    # the next 100 being another genre, etc. This generates an array
    # that looks like this:
    # [0, 0, 0,... , 0, 1, 1, 1, ...., 1, 2, ...., 2, ...]
    # It has 100 0s followed by  100 2s followed by 100 3s, and so on.
    # These are integer labels by genre.
    #           -berk, November 11, 2024

    genre_labels = numpy.repeat(numpy.arange(0, gtzan_genre_count), gtzan_songs_per_genre)

    genre_plot.set_title("Genre plot")
    genre_plot.set_xlabel("bpm")
    genre_plot.set_ylabel("spectral centroid")
    genre_plot.scatter(bpm, spectral_centroid, c=genre_labels, cmap="plasma", s=point_size)

    plt.show()

plot_gtzan()
