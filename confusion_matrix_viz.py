import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Values from the Apartment Leak Detection Performance image
confusion_data = np.array([[3577, 689],
                           [423, 5311]])

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Create blue-white colormap
colors = ['white', '#08306b']  # White to dark blue
n_bins = 100
cmap = LinearSegmentedColormap.from_list('blue_white', colors, N=n_bins)

# Normalize data for better color mapping
from matplotlib.colors import Normalize
norm = Normalize(vmin=confusion_data.min(), vmax=confusion_data.max())

# Create heatmap manually with imshow
im = ax.imshow(confusion_data, cmap=cmap, aspect='auto', norm=norm)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)

# Add text annotations with dynamic color based on background
for i in range(len(confusion_data)):
    for j in range(len(confusion_data[0])):
        value = confusion_data[i, j]
        # Use white text on dark backgrounds, black on light backgrounds
        text_color = 'white' if value > (confusion_data.max() + confusion_data.min()) / 2 else 'black'
        text = ax.text(j, i, int(value),
                      ha="center", va="center", color=text_color, fontsize=16, fontweight='bold')

# Set ticks and labels
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Pred Leak', 'Pred Normal'], fontsize=10)
ax.set_yticklabels(['Actual Leak', 'Actual Normal'], fontsize=10, rotation=90)

# Set y-axis label centered on left side
ax.set_ylabel('Actual Label', fontsize=11, rotation=90, labelpad=20)
ax.yaxis.set_label_coords(-0.08, 0.5)
ax.set_xlabel('Predicted Label', fontsize=11, labelpad=10)
ax.set_title('Confusion Matrix', fontsize=14, pad=15)

# Update tick label positions
ax.tick_params(axis='x', pad=8)
ax.tick_params(axis='y', pad=8)

plt.tight_layout()
plt.savefig('confusion_matrix_blue_white.png', dpi=300, bbox_inches='tight')
plt.show()

print("Confusion Matrix saved as 'confusion_matrix_blue_white.png'")

