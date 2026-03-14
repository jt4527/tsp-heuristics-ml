import matplotlib
matplotlib.use("Agg")  # No GUI needed

import matplotlib.pyplot as plt
import numpy as np
import requests

# Call your local API (make sure uvicorn is running)
resp = requests.get("http://localhost:8000/tsp_instance?seed=42&cities=20")
data = resp.json()
coords = np.array(data["coords"])
tours = data["tours"]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, tour_info) in enumerate(tours.items()):
    order = np.array(tour_info["order"])
    xs = np.append(coords[order, 0], coords[order[0], 0])
    ys = np.append(coords[order, 1], coords[order[0], 1])
    axs[i].plot(xs, ys, "-o", markersize=8)
    axs[i].set_title(f"{name}\n{tour_info['length']:.3f}")
    axs[i].axis("equal")

plt.tight_layout()
plt.savefig("tsp_comparison.png", dpi=300, bbox_inches="tight")
print("✅ tsp_comparison.png created!")