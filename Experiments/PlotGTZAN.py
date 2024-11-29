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

def find_gtzan_genres_most_associated_with_features(frame: pandas.DataFrame, plot: matplotlib.axes.Axes):
    # The only features we want to look at are the ones from the second index onwards.
    # The first two columns are filename and length (which is constant). We can ignore these.
    frame_feature_start_index = 2
    
    # The last column is the genre label, and there are 20 pairs of mfcc mean and variance.
    # As far as I can tell these become less meaningful the higher your mfcc index is.
    # So we will ignore the last 16, and because there are mean and variance for each one,
    # this becomes 32.
    frame_feature_end_index = frame.columns.size - (1 + 16*2)
    data = frame.to_numpy()[:, frame_feature_start_index:frame_feature_end_index].astype(np.float32)
    
    feature_count = data.shape[1]
    feature_genre_stdev = np.ndarray((feature_count, gtzan_genre_count), dtype=np.float32)
    # A [gtzan_genre_count, feature_count] array. Contains the standard deviation of each feature per genre.

    # ToDo: Figure out how to vectorize this
    for genre_index in range(gtzan_genre_count):
        start_index = genre_index * gtzan_songs_per_genre
        end_index = start_index + gtzan_songs_per_genre - 1
        for feature_index in range(feature_count):
            genre_feature = data[start_index:end_index, feature_index]
            # All the values of a specific feature per genre
            stdev = np.std(genre_feature)
            feature_genre_stdev[feature_index, genre_index] = stdev

    genres_ranked_per_feature = np.argsort(feature_genre_stdev, axis=1)

    genre_names = np.array([frame.at[i * gtzan_songs_per_genre, "label"] for i in range(gtzan_genre_count)])
    feature_names = np.array(frame.columns[frame_feature_start_index:frame_feature_end_index])

    cell_text = genre_names[genres_ranked_per_feature]

    plot.set_title(f"Genres most strongly associated with a given feature (Strongest = leftmost)")
    plot.axis("off") 
    table = plot.table(cellText=cell_text, rowLabels=feature_names, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(4)
    table.scale(1, 0.3)




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

    find_gtzan_genres_most_associated_with_features(frame, feature_stdev_plot)

    plt.show()

plot_gtzan()
