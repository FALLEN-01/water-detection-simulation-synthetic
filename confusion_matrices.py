import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Create blue-white colormap
colors = ['white', '#08306b']  # White to dark blue
n_bins = 100
cmap = LinearSegmentedColormap.from_list('blue_white', colors, N=n_bins)

def create_confusion_matrix(data, title, filename):
    """
    Create and save a confusion matrix visualization
    
    Parameters:
    - data: 2x2 numpy array with confusion matrix values
    - title: Title for the visualization
    - filename: Output filename for the image
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Normalize data for better color mapping
    norm = Normalize(vmin=data.min(), vmax=data.max())

    # Create heatmap manually with imshow
    im = ax.imshow(data, cmap=cmap, aspect='auto', norm=norm)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)

    # Add text annotations with dynamic color based on background
    for i in range(len(data)):
        for j in range(len(data[0])):
            value = data[i, j]
            # Use white text on dark backgrounds, black on light backgrounds
            text_color = 'white' if value > (data.max() + data.min()) / 2 else 'black'
            text = ax.text(j, i, int(value),
                          ha="center", va="center", color=text_color, fontsize=16, fontweight='bold')

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Leak', 'Pred Normal'], fontsize=10)
    ax.set_yticklabels(['Actual Leak', 'Actual Normal\nLabel'], fontsize=10, rotation=90)

    # Add grid lines
    for i in range(len(data) + 1):
        ax.axhline(i - 0.5, color='black', linewidth=2)
    for j in range(len(data[0]) + 1):
        ax.axvline(j - 0.5, color='black', linewidth=2)

    # Set y-axis label centered on left side
    ax.set_ylabel('Actual Label', fontsize=11, rotation=90, labelpad=20)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"{title} confusion matrix saved as '{filename}'")

# Apartment confusion matrix values
apartment_data = np.array([[3577, 689],
                           [423, 5311]])

# Household confusion matrix values
household_data = np.array([[171, 30],
                           [4, 98853]])

# Generate both confusion matrices
create_confusion_matrix(apartment_data, 'Apartment', 'apartment_confusion_matrix.png')
create_confusion_matrix(household_data, 'Household', 'household_confusion_matrix.png')
