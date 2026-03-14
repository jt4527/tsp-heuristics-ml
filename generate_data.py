import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import json
import os

def nearest_neighbour(coords, dists):
    """Greedy nearest neighbour tour starting from city 0."""
    n = len(coords)
    tour = [0]
    visited = set([0])
    while len(tour) < n:
        current = tour[-1]
        unvisited = [i for i in range(n) if i not in visited]
        next_city = min(unvisited, key=lambda i: dists[current, i])
        tour.append(next_city)
        visited.add(next_city)
    return np.array(tour)

def tour_length(tour, dists):
    """Total length of tour (including return to start)."""
    return sum(dists[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))

def two_opt(coords, dists, tour, max_iters=100):
    """2-opt local search to improve initial tour."""
    tour = list(tour)
    best_length = tour_length(tour, dists)
    
    for iteration in range(max_iters):
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 2, len(tour)):
                # Reverse segment i to j-1
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_length = tour_length(new_tour, dists)
                
                if new_length < best_length:
                    tour = new_tour
                    best_length = new_length
                    improved = True
        
        if not improved:
            break
    
    return np.array(tour)

def generate_tsp_instance(n_cities=20):
    """Generate random TSP instance: n_cities in unit square."""
    return np.random.rand(n_cities, 2)

def dist_matrix(coords):
    """Euclidean distance matrix."""
    return cdist(coords, coords, 'euclidean')

def create_training_examples(coords, tour, n_cities):
    """Create (state, next_city) pairs from oracle tour.
    
    State vector (fixed size 97):
    - current one-hot (20)
    - visited mask (20) 
    - unvisited coords (19x2=38, padded)
    - dists from current to unvisited (19, padded)
    """
    examples = []
    for start_pos in range(n_cities):
        current = tour[start_pos]
        visited_mask = np.zeros(n_cities)
        visited_mask[tour[:start_pos+1]] = 1.0
        unvisited_indices = np.where(visited_mask == 0)[0]
        
        # Fixed-size unvisited features (pad with zeros if <19 unvisited)
        n_unvis = len(unvisited_indices)
        unvis_coords = np.zeros((19, 2))
        unvis_coords[:n_unvis] = coords[unvisited_indices]
        unvis_dists = np.zeros(19)
        unvis_dists[:n_unvis] = dist_matrix(coords)[current][unvisited_indices]
        
        state = np.concatenate([
            np.eye(n_cities)[current],      # (20,)
            visited_mask,                   # (20,)
            unvis_coords.flatten(),         # (38,)
            unvis_dists                     # (19,)
        ]).astype(np.float32)               # Total: 97 dims
        
        next_city = tour[(start_pos + 1) % n_cities]
        examples.append({
            "state": state.tolist(), 
            "next": int(next_city)
        })
    return examples

# ========================================
# MAIN EXECUTION BLOCK (functions defined above)
# ========================================
if __name__ == "__main__":
    # Config - scale up as needed
    n_instances_train = 100   # or 10000 for full dataset
    n_instances_eval = 10
    n_cities = 20
    
    for split, n_inst in [('train', n_instances_train), ('eval', n_instances_eval)]:
        os.makedirs(f'data/{split}_instances', exist_ok=True)
        
        for i in tqdm(range(n_inst), desc=f"Generating {split}"):
            # 1. Generate instance
            coords = generate_tsp_instance(n_cities)
            dists = dist_matrix(coords)
            
            # 2. Nearest neighbour (baseline)
            nn_tour = nearest_neighbour(coords, dists)
            
            # 3. 2-opt improvement (oracle for imitation)
            opt_tour = two_opt(coords, dists, nn_tour)
            
            # 4. Extract training examples from oracle tour
            examples = create_training_examples(coords, opt_tour, n_cities)
            
            # 5. Save everything
            data = {
                "coords": coords.tolist(),
                "nn_tour": nn_tour.tolist(),
                "opt_tour": opt_tour.tolist(),
                "tour_length_nn": float(tour_length(nn_tour, dists)),
                "tour_length_opt": float(tour_length(opt_tour, dists)),
                "examples": examples
            }
            with open(f'data/{split}_instances/instance_{i:04d}.json', 'w') as f:
                json.dump(data, f, indent=2)
    
    print("✅ Generation complete!")
    print(f"Created {n_instances_train} train + {n_instances_eval} eval instances")
    print("Files saved in data/train_instances/ and data/eval_instances/")
    print("Next step: python train.py")
