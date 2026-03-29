import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

# Apartment confusion matrix values
apartment_data = np.array([[3577, 689],
                           [423, 5311]])

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create blue-white colormap
colors = ['white', '#08306b']  # White to dark blue
n_bins = 100
cmap = LinearSegmentedColormap.from_list('blue_white', colors, N=n_bins)

# Normalize data for better color mapping
norm = Normalize(vmin=apartment_data.min(), vmax=apartment_data.max())

# Create heatmap manually with imshow
im = ax.imshow(apartment_data, cmap=cmap, aspect='auto', norm=norm)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)

# Add text annotations with dynamic color based on background
for i in range(len(apartment_data)):
    for j in range(len(apartment_data[0])):
        value = apartment_data[i, j]
        # Use white text on dark backgrounds, black on light backgrounds
        text_color = 'white' if value > (apartment_data.max() + apartment_data.min()) / 2 else 'black'
        text = ax.text(j, i, int(value),
                      ha="center", va="center", color=text_color, fontsize=16, fontweight='bold')

# Set ticks and labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred Leak', 'Pred Normal'], fontsize=10)
ax.set_yticklabels(['Actual Leak', 'Actual Normal\nLabel'], fontsize=10, rotation=90)

# Add grid lines
ax.set_xticks([0.5, 1.5], minor=False)
ax.set_yticks([0.5, 1.5], minor=False)
ax.grid(which='both', color='black', linewidth=2)

# Set y-axis label centered on left side
ax.set_ylabel('Actual Label', fontsize=11, rotation=90, labelpad=20)
ax.yaxis.set_label_coords(-0.08, 0.5)
ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)

plt.tight_layout()
plt.savefig('apartment_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("Apartment confusion matrix saved as 'apartment_confusion_matrix.png'")
