import numpy
import k_means_constrained
import pandas


#
# Test clustering features in respect to genre
#


genre_count = 10
songs_per_genre = 100

graph_point_size = 10
graph_cluster_label_size = 100


def gtzan_k_means_constrained():
    return k_means_constrained.KMeansConstrained(
        n_clusters = genre_count,
        size_min = int(songs_per_genre * 2/3),
        size_max = int(songs_per_genre * 1.5),
        random_state = 0
    )

def gtzan_load():
    frame = pandas.read_csv("Datasets/GTZAN/features_30_sec.csv")
    
    # The first two features are:
    # The file name (which we don't need)
    # The sample length (which is always the same).
    # So we can ignore them.
    data = frame.to_numpy()[:, 2:frame.columns.size-1].astype(numpy.float32)
    feature_names = numpy.array(frame.columns[2:frame.columns.size-1])
    genre_names = numpy.array([frame.at[i * songs_per_genre, "label"] for i in range(genre_count)])
    genre_labels = numpy.repeat(numpy.arange(genre_count), songs_per_genre)

    return (frame, data, feature_names, genre_names, genre_labels)

