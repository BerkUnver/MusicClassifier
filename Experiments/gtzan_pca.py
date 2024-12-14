import gtzan
import numpy
import torch
import matplotlib
import matplotlib.pyplot as plt

def gtzan_torch_pca(
    data: numpy.array,
    genre_names: numpy.array,
    feature_names: numpy.array,
    genre_labels: numpy.array,
    cluster_plot: matplotlib.axes.Axes,
    genre_plot: matplotlib.axes.Axes):
    
    tensor = torch.Tensor(data)
    tensor = torch.nn.functional.normalize(tensor, dim=1)

    (_, _, principal_components) = torch.pca_lowrank(tensor)
    
    components = torch.matmul(tensor, principal_components[:, :2]).numpy()

    clf = gtzan.gtzan_k_means_constrained()
    labels = clf.fit_predict(components)

    components_x = components[:, 0]
    components_y = components[:, 1]

    cluster_plot.set_title("PCA cluster plot")
    cluster_plot.set_xlabel("Component with the greatest variance")
    cluster_plot.set_ylabel("Component with the second-greatest variance")

    cluster_plot.scatter(components_x, components_y, c=labels, cmap="plasma", s=gtzan.graph_point_size)
    cluster_plot.scatter(clf.cluster_centers_[:, 0], clf.cluster_centers_[:, 1], marker="X", s=gtzan.graph_cluster_label_size)
    
    genre_plot.set_title("PCA genre plot")
    genre_plot.set_xlabel("Component with the greatest variance")
    genre_plot.set_ylabel("Component with the second-greatest variance")
    genre_plot.scatter(components_x, components_y, c=genre_labels, cmap="plasma", s=gtzan.graph_point_size)

    colors = plt.colormaps['plasma'](numpy.linspace(0, 1, gtzan.genre_count))
    patches = [matplotlib.patches.Patch(color=color) for color in colors]
    genre_plot.legend(patches, genre_names)


_, (cluster_plot, genre_plot) = plt.subplots(ncols=2)
_, data, feature_names, genre_names, genre_labels = gtzan.gtzan_load()
gtzan_torch_pca(data, genre_names, feature_names, genre_labels, cluster_plot, genre_plot)
plt.show()
