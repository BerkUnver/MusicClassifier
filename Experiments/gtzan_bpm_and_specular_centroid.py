import gtzan
import numpy
import matplotlib
import matplotlib.pyplot as plt

def gtzan_bpm_and_specular_centroid_unnormalized(
    bpm: numpy.array,
    spectral_centroid: numpy.array,
    plot: matplotlib.axes.Axes):

    clf = gtzan.gtzan_k_means_constrained()

    data = numpy.stack((bpm, spectral_centroid), axis=-1)
    labels = clf.fit_predict(data)
    
    plot.set_title("Cluster plot")
    plot.set_xlabel("bpm")
    plot.set_ylabel("spectral centroid")
    plot.scatter(bpm, spectral_centroid, c=labels, cmap="plasma", s=gtzan.graph_point_size)
    plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=gtzan.graph_cluster_label_size)
    

# bpm and spectral centroids are in drastically different units.
# This results in the clustering being very linear, as seen in
# the non-normalized graph. To fix this, we will normalize the 
# bpm and spectral centroid to be between 0 and 1, and then cluster.
#           -berk, November 11, 2024
def gtzan_bpm_and_specular_centroid_normalized(
    bpm: numpy.array,
    spectral_centroid: numpy.array,
    plot: matplotlib.axes.Axes):

    def numpy_normalize(array):
        array_min = numpy.min(array)
        array_max = numpy.max(array)
        return (array - array_min) / (array_max - array_min)

    clf = gtzan.gtzan_k_means_constrained()

    normalized_bpm = numpy_normalize(bpm)
    normalized_spectral_centroid = numpy_normalize(spectral_centroid)
    data = numpy.stack((normalized_bpm, normalized_spectral_centroid), axis=-1)
    labels = clf.fit_predict(data)

    plot.set_title("Normalized cluster plot")
    plot.set_xlabel("normalized bpm")
    plot.set_ylabel("normalized spectral centroid")

    plot.scatter(normalized_bpm, normalized_spectral_centroid, c=labels, cmap="plasma", s=gtzan.graph_point_size)
    plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=gtzan.graph_cluster_label_size)


def gtzan_bpm_and_specular_centroid_genres(
    bpm: numpy.array,
    spectral_centroid: numpy.array,
    genre_names: numpy.array,
    genre_labels: numpy.array,
    plot: matplotlib.axes.Axes):

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

    plot.set_title("Genre plot")
    plot.set_xlabel("bpm")
    plot.set_ylabel("spectral centroid")
    plot.scatter(bpm, spectral_centroid, c=genre_labels, cmap="plasma", s=gtzan.graph_point_size)
    colors = plt.colormaps['plasma'](numpy.linspace(0, 1, gtzan.genre_count))
    patches = [matplotlib.patches.Patch(color=color) for color in colors]
    plot.legend(patches, genre_names)    


_, (unnormalized_plot, normalized_plot, genre_plot) = plt.subplots(ncols=3)

frame, data, feature_names, genre_names, genre_labels = gtzan.gtzan_load()
bpm = frame["tempo"]
spectral_centroid = frame["spectral_centroid_mean"]

gtzan_bpm_and_specular_centroid_unnormalized(bpm, spectral_centroid, unnormalized_plot)
gtzan_bpm_and_specular_centroid_normalized(bpm, spectral_centroid, normalized_plot)
gtzan_bpm_and_specular_centroid_genres(bpm, spectral_centroid, genre_names, genre_labels, genre_plot)

plt.show()


