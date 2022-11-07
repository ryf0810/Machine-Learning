import numpy as np

def plot_decision_boundary(clf, xlim, ylim, grid_resolution):
    """Display how clf classifies each point in the space specified by xlim and ylim.
    
    - clf is a classifier.
    - xlim and ylim are each 2-tuples of the form (low, high).
    - grid_resolution specifies the number of points into which the xlim is divided
      and the number into which the ylim interval is divided. The function plots
      grid_resolution * grid_resolution points."""

    # ... your code here ...
    x_grid_points = np.tile(np.linspace(xlim[0], xlim[1], grid_resolution), grid_resolution)
    y_grid_points = np.repeat(np.linspace(ylim[0], ylim[1], grid_resolution), grid_resolution)
    coordinates = np.stack((x_grid_points, y_grid_points),axis=1) # size:(10000, 2)
    colors = {-1:'pink', 1:'lightskyblue'}
    predicted_labels = clf.predict(coordinates)
    # convert predicted labels into color (string)
    padding = []
    for label in predicted_labels:
      padding.append(colors[label])
    plt.scatter(x_grid_points, y_grid_points, c=np.asarray(padding))
