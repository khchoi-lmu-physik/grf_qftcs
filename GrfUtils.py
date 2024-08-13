
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def create_colorbar(min_val, max_val, cmap, filename, resolution=17):
    # Create a dummy ScalarMappable with the colormap
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create the colorbar
    fig = plt.figure(figsize=(resolution, resolution))
    cbar = plt.colorbar(sm)

    cbar.ax.tick_params(axis='both', colors='black', direction='out', length=32, width=5, labelsize=60)
    fig.savefig(filename, bbox_inches = 'tight',pad_inches = 0.2)
    plt.close(fig)

def create_log_colorbar(min_val, max_val, cmap, filename, resolution=17):

    # Use LogNorm for logarithmic scaling
    norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Create the colorbar
    fig = plt.figure(figsize=(resolution, resolution))
    cbar = plt.colorbar(sm)

    cbar.ax.tick_params(axis='both', colors='black', direction='out', length=32, width=5, labelsize=60)
    fig.savefig(filename, bbox_inches = 'tight',pad_inches = 0.2)
    plt.close(fig)
