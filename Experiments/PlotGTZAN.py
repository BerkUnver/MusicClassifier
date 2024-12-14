import numpy as np
import sklearn.decomposition
import k_means_constrained
import matplotlib
import matplotlib.pyplot as plt
import pandas
import torch


#
# Test clustering features in respect to genre
#


gtzan_genre_count = 10
gtzan_songs_per_genre = 100


graph_point_size = 10
graph_cluster_label_size = 100

def gtzan_k_means_constrained():
    return k_means_constrained.KMeansConstrained(
        n_clusters = gtzan_genre_count,
        size_min = int(gtzan_songs_per_genre * 2/3),
        size_max = int(gtzan_songs_per_genre * 1.5),
        random_state = 0
    )

def gtzan_bpm_and_specular_centroid_unnormalized(
    bpm: np.array,
    spectral_centroid: np.array,
    plot: matplotlib.axes.Axes):

    clf = gtzan_k_means_constrained()

    data = np.stack((bpm, spectral_centroid), axis=-1)
    labels = clf.fit_predict(data)
    
    plot.set_title("Cluster plot")
    plot.set_xlabel("bpm")
    plot.set_ylabel("spectral centroid")
    plot.scatter(bpm, spectral_centroid, c=labels, cmap="plasma", s=graph_point_size)
    plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=graph_cluster_label_size)
    

# bpm and spectral centroids are in drastically different units.
# This results in the clustering being very linear, as seen in
# the non-normalized graph. To fix this, we will normalize the 
# bpm and spectral centroid to be between 0 and 1, and then cluster.
#           -berk, November 11, 2024
def gtzan_bpm_and_specular_centroid_normalized(
    bpm: np.array,
    spectral_centroid: np.array,
    plot: matplotlib.axes.Axes):

    def numpy_normalize(array):
        array_min = np.min(array)
        array_max = np.max(array)
        return (array - array_min) / (array_max - array_min)

    clf = gtzan_k_means_constrained()

    normalized_bpm = numpy_normalize(bpm)
    normalized_spectral_centroid = numpy_normalize(spectral_centroid)
    data = np.stack((normalized_bpm, normalized_spectral_centroid), axis=-1)
    labels = clf.fit_predict(data)

    plot.set_title("Normalized cluster plot")
    plot.set_xlabel("normalized bpm")
    plot.set_ylabel("normalized spectral centroid")

    plot.scatter(normalized_bpm, normalized_spectral_centroid, c=labels, cmap="plasma", s=graph_point_size)
    plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=graph_cluster_label_size)


def gtzan_bpm_and_specular_centroid_genres(
    bpm: np.array,
    spectral_centroid: np.array,
    genre_names: np.array,
    genre_labels: np.array,
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
    plot.scatter(bpm, spectral_centroid, c=genre_labels, cmap="plasma", s=graph_point_size)
    colors = plt.colormaps['plasma'](np.linspace(0, 1, gtzan_genre_count))
    patches = [matplotlib.patches.Patch(color=color) for color in colors]
    plot.legend(patches, genre_names)    


def gtzan_genres_most_associated_with_features(
    data: np.array,
    genre_names: np.array,
    feature_names: np.array,
    genre_plot: matplotlib.axes.Axes,
    association_plot: matplotlib.axes.Axes):

    # The last 40 features in the input data are 20 pairs of mfcc means and variances.
    # As far as I can tell these become less meaningful the higher your mfcc index is.
    # So we will ignore the last 16, and because there are mean and variance for each one,
    # this becomes 32.
    feature_end_index = data.shape[1] - 32
    data = data[:, :feature_end_index]
    feature_names = feature_names[:feature_end_index]


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

    genres_ranked_per_feature = genre_names[np.argsort(feature_genre_stdev, axis=1)]

    association_ranked_per_feature = np.sort(feature_genre_stdev, axis=1)
    # Let's normalize these, using the first entry of each row as the 1 value.
    association_normalization = association_ranked_per_feature[:, 0][:, np.newaxis].repeat(association_ranked_per_feature.shape[1], axis=1)
    association_ranked_per_feature /= association_normalization

    genre_plot.set_title("Genres most strongly associated with a given feature (Strongest = leftmost)")
    genre_plot.axis("off") 
    genre_table = genre_plot.table(cellText=genres_ranked_per_feature, rowLabels=feature_names, loc="center")
    genre_table.auto_set_font_size(False)
    genre_table.set_fontsize(4)
    genre_table.scale(1, 0.3)

    association_plot.set_title("Amount each genre is associated by")
    association_plot.axis("off")
    association_table = association_plot.table(cellText=association_ranked_per_feature, rowLabels=feature_names, loc="center")
    association_table.auto_set_font_size(False)
    association_table.set_fontsize(4)
    association_table.scale(1, 0.3)



def gtzan_pca(
    data: np.array,
    genre_names: np.array,
    feature_names: np.array,
    genre_labels: np.array,
    cluster_plot: matplotlib.axes.Axes,
    genre_plot: matplotlib.axes.Axes):
    
    pca = sklearn.decomposition.PCA(n_components=2)

    # TODO: Fix having to transpose, then untranspose it!
    pca.fit(data.T)

    clf = gtzan_k_means_constrained()

    components = pca.components_.T
    labels = clf.fit_predict(components)

    components_x = components[:, 0]
    components_y = components[:, 1]

    cluster_plot.set_title("PCA cluster plot")
    cluster_plot.set_xlabel("Component with the greatest variance")
    cluster_plot.set_ylabel("Component with the second-greatest variance")

    cluster_plot.scatter(components_x, components_y, c=labels, cmap="plasma", s=graph_point_size)
    cluster_plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=graph_cluster_label_size)

    genre_plot.set_title("PCA genre plot")
    genre_plot.set_xlabel("Component with the greatest variance")
    genre_plot.set_ylabel("Component with the second-greatest variance")
    genre_plot.scatter(components_x, components_y, c=genre_labels, cmap="plasma", s=graph_point_size)


def gtzan_torch_pca(
    data: np.array,
    genre_names: np.array,
    feature_names: np.array,
    genre_labels: np.array,
    cluster_plot: matplotlib.axes.Axes,
    genre_plot: matplotlib.axes.Axes):
    
    tensor = torch.Tensor(data)
    tensor = torch.nn.functional.normalize(tensor, dim=1)

    (_, _, principal_components) = torch.pca_lowrank(tensor)
    
    components = torch.matmul(tensor, principal_components[:, :2]).numpy()

    clf = gtzan_k_means_constrained()
    labels = clf.fit_predict(components)

    components_x = components[:, 0]
    components_y = components[:, 1]

    cluster_plot.set_title("PCA cluster plot")
    cluster_plot.set_xlabel("Component with the greatest variance")
    cluster_plot.set_ylabel("Component with the second-greatest variance")

    cluster_plot.scatter(components_x, components_y, c=labels, cmap="plasma", s=graph_point_size)
    cluster_plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=graph_cluster_label_size)
    
    genre_plot.set_title("PCA genre plot")
    genre_plot.set_xlabel("Component with the greatest variance")
    genre_plot.set_ylabel("Component with the second-greatest variance")
    genre_plot.scatter(components_x, components_y, c=genre_labels, cmap="plasma", s=graph_point_size)

    colors = plt.colormaps['plasma'](np.linspace(0, 1, gtzan_genre_count))
    patches = [matplotlib.patches.Patch(color=color) for color in colors]
    genre_plot.legend(patches, genre_names)



def plot_gtzan():
    frame = pandas.read_csv("Datasets/GTZAN/features_30_sec.csv")
    
    # The first two features are:
    # The file name (which we don't need)
    # The sample length (which is always the same).
    # So we can ignore them.
    data = frame.to_numpy()[:, 2:frame.columns.size-1].astype(np.float32)
    feature_names = np.array(frame.columns[2:frame.columns.size-1])
    genre_names = np.array([frame.at[i * gtzan_songs_per_genre, "label"] for i in range(gtzan_genre_count)])
    genre_labels = np.repeat(np.arange(gtzan_genre_count), gtzan_songs_per_genre)

    _, cluster_plot = plt.subplots()

    # _, ((cluster_plot, normalized_plot), (genre_plot, feature_stdev_plot), (pca_cluster_plot, pca_genre_plot)) = plt.subplots(ncols=2, nrows=3)

    bpm = frame["tempo"]
    specular_centroid = frame["spectral_centroid_mean"]
    
    gtzan_bpm_and_specular_centroid_unnormalized(bpm, specular_centroid, cluster_plot)
    # gtzan_bpm_and_specular_centroid_normalized(bpm, specular_centroid, normalized_plot)
    # gtzan_bpm_and_specular_centroid_genres(bpm, specular_centroid, genre_names, genre_labels, genre_plot)
    # gtzan_genres_most_associated_with_features(data, genre_names, feature_names, genre_plot, association_plot)
    # gtzan_torch_pca(data, genre_names, feature_names, genre_labels, cluster_plot, genre_plot)

    plt.show()

plot_gtzan()
