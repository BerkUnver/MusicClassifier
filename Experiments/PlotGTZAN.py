import numpy as np
import k_means_constrained
import matplotlib
import matplotlib.pyplot as plt
import pandas


#
# Test clustering features in respect to genre
#

# This is just a basic example to show how we can do k-means clustering 
# on a per-feature basis. A future step we can take is to figure out
# which of the gtzan pre-extracted features most closely correspond to 
# genre, which may have some interesting results. I'm not sure if that 
# will involve deep learning or if we can do that with more 
# conventional algorithms.
#           -berk, November 15, 2024



gtzan_genre_count = 10
gtzan_songs_per_genre = 100


def find_gtzan_features_most_associated_with_genre(frame: pandas.DataFrame, plot: matplotlib.axes.Axes):
    # The only features we want to look at are the ones from the second index onwards.
    # The first two columns are filename and length (which is constant), and the last column is the genre as a string name. We can ignore these.
    frame_feature_start_index = 2
    data = frame.to_numpy()[:, frame_feature_start_index:frame.columns.size-1].astype(np.float32)
    
    feature_count = data.shape[1]
    genre_feature_stdev = np.ndarray((gtzan_genre_count, feature_count), dtype=np.float32)
    # A [gtzan_genre_count, feature_count] array. Contains the standard deviation of each feature per genre.

    # ToDo: Figure out how to vectorize this
    for genre_index in range(gtzan_genre_count):
        start_index = genre_index * gtzan_songs_per_genre
        end_index = start_index + gtzan_songs_per_genre - 1
        for feature_index in range(feature_count):
            genre_feature = data[start_index:end_index, feature_index]
            # All the values of a specific feature per genre

            mean = np.mean(genre_feature)
            normalization_factor = 1 / mean # Multiply the mean by this to get 1. Used to normalize the standard deviation.
            stdev = np.std(genre_feature, mean=mean)
            genre_feature_stdev[genre_index, feature_index] = stdev * normalization_factor
    
    
    stdev_mins = np.argmin(genre_feature_stdev, axis=0)
    # A [feature_count] array with the indices of the genre most closely associated (lowest stdev) with a given feature.
    
    plot.set_title("Genres most closely associated with features (INCOMPLETE!)")
    plot.set_xlabel("Genre")
    plot.set_ylabel("Feature standard deviation")
    plot.set_yscale("log")
    
    genres = frame["label"][::gtzan_songs_per_genre]
    plot.set_xticks(np.arange(gtzan_genre_count), genres)

    for genre_index in range(gtzan_genre_count):
        feature_stdev = genre_feature_stdev[genre_index]
        for feature_index in range(feature_stdev.size):
            stdev = feature_stdev[feature_index]
            feature_name = frame.columns[frame_feature_start_index + feature_index]
            plot.scatter(genre_index, stdev, c="#0000ff")
            plot.annotate(feature_name, (genre_index, stdev))




def plot_gtzan():
    frame = pandas.read_csv("Datasets/GTZAN/features_30_sec.csv")
    
    #
    # Generate the plots that cluster based on spectral centroid and bpm.
    #

    try:
        bpm = frame["tempo"]
        spectral_centroid = frame["spectral_centroid_mean"]
    except ValueError:
        assert False

    # No spectral flatness in the GTZAN pre-extracted features so we will avoid that for now.
    

    clf = k_means_constrained.KMeansConstrained(
        n_clusters = gtzan_genre_count,
        size_min = int(gtzan_songs_per_genre * 2/3),
        size_max = int(gtzan_songs_per_genre * 1.5),
        random_state = 0
    )
    
    cluster_data = np.stack((bpm, spectral_centroid), axis=-1)
    cluster_labels = clf.fit_predict(cluster_data)
    
    _, ((cluster_plot, normalized_plot), (genre_plot, feature_stdev_plot)) = plt.subplots(ncols=2, nrows=2)
    
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
        array_min = np.min(array)
        array_max = np.max(array)
        return (array - array_min) / (array_max - array_min)

    normalized_bpm = numpy_normalize(bpm)
    normalized_spectral_centroid = numpy_normalize(spectral_centroid)
    normalized_data = np.stack((normalized_bpm, normalized_spectral_centroid), axis=-1)
    normalized_labels = clf.fit_predict(normalized_data)

    normalized_plot.set_title("Normalized cluster plot")
    normalized_plot.set_xlabel("normalized bpm")
    normalized_plot.set_ylabel("normalized spectral centroid")

    normalized_plot.scatter(normalized_bpm, normalized_spectral_centroid, c=normalized_labels, cmap="plasma", s=point_size)
    normalized_plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=label_size)

    
    #
    # Now, generate a plot where each point on the (bpm, spectral_centroid) axis is a color that corresponds to its genre.
    #

    # The songs are in order, with the first 100 songs being one genre,
    # the next 100 being another genre, etc. This generates an array
    # that looks like this:
    # [0, 0, 0,... , 0, 1, 1, 1, ...., 1, 2, ...., 2, ...]
    # It has 100 0s followed by  100 2s followed by 100 3s, and so on.
    # These are integer labels by genre.
    #           -berk, November 11, 2024
    genre_labels = np.repeat(np.arange(0, gtzan_genre_count), gtzan_songs_per_genre)

    genre_plot.set_title("Genre plot")
    genre_plot.set_xlabel("bpm")
    genre_plot.set_ylabel("spectral centroid")
    genre_plot.scatter(bpm, spectral_centroid, c=genre_labels, cmap="plasma", s=point_size)

    find_gtzan_features_most_associated_with_genre(frame, feature_stdev_plot)

    plt.show()

plot_gtzan()
