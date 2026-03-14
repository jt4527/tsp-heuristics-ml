import matplotlib.pyplot as plt
import numpy as np
import json

# Load your JSON
with open('tsp_new.json', 'r') as f:
    data = json.load(f)

coords = np.array(data["coords"])
tours = data["tours"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, tour_info) in enumerate(tours.items()):
    order = np.array(tour_info["order"])
    
    # Plot tour (close loop)
    xs = np.append(coords[order, 0], coords[order[0], 0])
    ys = np.append(coords[order, 1], coords[order[0], 1])
    
    axs[i].plot(xs, ys, "-o", markersize=8, linewidth=2)
    axs[i].scatter(coords[:, 0], coords[:, 1], c='lightgray', s=100, alpha=0.7)
    axs[i].set_title(f"{name.replace('_', ' ').title()}\nLength: {tour_info['length']:.3f}", fontsize=12)
    axs[i].axis("equal")
    axs[i].grid(True, alpha=0.3)

plt.suptitle("TSP Tours Comparison (seed=42)", fontsize=14)
plt.tight_layout()
plt.savefig("tsp_comparison.png", dpi=300, bbox_inches='tight')  # Portfolio screenshot!
plt.show()
