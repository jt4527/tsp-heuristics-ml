import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI
from scipy.spatial.distance import cdist

# ==== Reuse from your project ====

class TSPPolicy(nn.Module):
    def __init__(self, input_dim=97, n_cities=20, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_cities)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def dist_matrix(coords):
    return cdist(coords, coords, 'euclidean')

def tour_length(tour, dists):
    return float(sum(dists[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour))))

def nearest_neighbour(coords, dists):
    n = len(coords)
    tour = [0]
    visited = set([0])
    while len(tour) < n:
        current = tour[-1]
        unvisited = [i for i in range(n) if i not in visited]
        next_city = min(unvisited, key=lambda i: dists[current, i])
        tour.append(next_city)
        visited.add(next_city)
    return np.array(tour, dtype=int)

def two_opt(coords, dists, tour, max_iters=100):
    tour = list(tour)
    best_length = tour_length(tour, dists)
    for _ in range(max_iters):
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 2, len(tour)):
                new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                new_length = tour_length(new_tour, dists)
                if new_length < best_length:
                    tour = new_tour
                    best_length = new_length
                    improved = True
        if not improved:
            break
    return np.array(tour, dtype=int)

def build_greedy_tour(model, device, coords, n_cities):
    model.eval()
    dists = dist_matrix(coords)
    tour = [0]
    visited_mask = np.zeros(n_cities)
    visited_mask[0] = 1.0
    with torch.no_grad():
        for _ in range(1, n_cities):
            current = tour[-1]
            unvisited_indices = np.where(visited_mask == 0)[0]
            n_unvis = len(unvisited_indices)
            unvis_coords = np.zeros((19, 2))
            unvis_coords[:n_unvis] = coords[unvisited_indices]
            unvis_dists = np.zeros(19)
            unvis_dists[:n_unvis] = dists[current][unvisited_indices]
            state = np.concatenate([
                np.eye(n_cities)[current],
                visited_mask,
                unvis_coords.flatten(),
                unvis_dists
            ]).astype(np.float32)
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = model(state_t)
            unvis_logits = logits[0][unvisited_indices]
            next_idx = unvis_logits.argmax().item()
            next_city = int(unvisited_indices[next_idx])
            tour.append(next_city)
            visited_mask[next_city] = 1.0
    return np.array(tour, dtype=int)

# ==== FastAPI app ====

app = FastAPI(title="TSP Heuristic Demo")

device = torch.device("cpu")
n_cities = 20

model = TSPPolicy().to(device)
model.load_state_dict(torch.load("models/tsp_policy.pt", map_location=device))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tsp_instance")
def tsp_instance(seed: int = 0, cities: int = 20):
    np.random.seed(seed)
    coords = np.random.rand(cities, 2)
    dists = dist_matrix(coords)

    nn_tour = nearest_neighbour(coords, dists)
    nn_len = tour_length(nn_tour, dists)

    opt_tour = two_opt(coords, dists, nn_tour)
    opt_len = tour_length(opt_tour, dists)

    learned_tour = build_greedy_tour(model, device, coords, cities)
    learned_len = tour_length(learned_tour, dists)

    return {
        "coords": coords.tolist(),
        "tours": {
            "nearest_neighbour": {
                "order": nn_tour.tolist(),
                "length": nn_len
            },
            "learned": {
                "order": learned_tour.tolist(),
                "length": learned_len
            },
            "two_opt": {
                "order": opt_tour.tolist(),
                "length": opt_len
            }
        }
    }
