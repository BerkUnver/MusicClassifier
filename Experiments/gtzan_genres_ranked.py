import gtzan
import numpy
import matplotlib
import matplotlib.pyplot as plt

def gtzan_genres_most_associated_with_features(
    data: numpy.array,
    genre_names: numpy.array,
    feature_names: numpy.array,
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
    feature_genre_stdev = numpy.ndarray((feature_count, gtzan.genre_count), dtype=numpy.float32)
    # A [gtzan.genre_count, feature_count] array. Contains the standard deviation of each feature per genre.

    # ToDo: Figure out how to vectorize this
    for genre_index in range(gtzan.genre_count):
        start_index = genre_index * gtzan.songs_per_genre
        end_index = start_index + gtzan.songs_per_genre - 1
        for feature_index in range(feature_count):
            genre_feature = data[start_index:end_index, feature_index]
            # All the values of a specific feature per genre
            stdev = numpy.std(genre_feature)
            feature_genre_stdev[feature_index, genre_index] = stdev

    genres_ranked_per_feature = genre_names[numpy.argsort(feature_genre_stdev, axis=1)]

    association_ranked_per_feature = numpy.sort(feature_genre_stdev, axis=1)
    # Let's normalize these, using the first entry of each row as the 1 value.
    association_normalization = association_ranked_per_feature[:, 0][:, numpy.newaxis].repeat(association_ranked_per_feature.shape[1], axis=1)
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


_, (feature_plot, association_plot) = plt.subplots(nrows=2)

_, data, feature_names, genre_names, _ = gtzan.gtzan_load()

gtzan_genres_most_associated_with_features(data, genre_names, feature_names, feature_plot, association_plot)

plt.show()
