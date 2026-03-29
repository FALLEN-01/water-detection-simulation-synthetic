import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Apartment confusion matrix values
apartment_data = np.array([[3577, 689],
                           [423, 5311]])

# Household confusion matrix values
household_data = np.array([[171, 30],
                           [4, 98853]])

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Create blue-white colormap
colors = ['white', '#08306b']  # White to dark blue
n_bins = 100
cmap = LinearSegmentedColormap.from_list('blue_white', colors, N=n_bins)

# Function to create confusion matrix
def create_confusion_matrix(ax, data, title):
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
    
    # Set y-axis label centered on left side
    ax.set_ylabel('Actual Label', fontsize=11, rotation=90, labelpad=20)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)

# Create apartment matrix
create_confusion_matrix(ax1, apartment_data, 'Apartment')
ax1.text(0.5, -0.25, 'Apartment', transform=ax1.transAxes, fontsize=12, ha='center', fontweight='bold')

# Create household matrix
create_confusion_matrix(ax2, household_data, 'Household')
ax2.text(0.5, -0.25, 'Household', transform=ax2.transAxes, fontsize=12, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparison confusion matrices saved as 'confusion_matrices_comparison.png'")
