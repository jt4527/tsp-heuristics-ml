import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from scipy.spatial.distance import cdist
import os
from tqdm import tqdm

# Model definition (same as train.py)
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
    return np.array(tour)

def tour_length(tour, dists):
    return sum(dists[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour)))

def build_greedy_tour(model, device, coords, n_cities):
    model.eval()
    dists = dist_matrix(coords)
    tour = [0]
    visited_mask = np.zeros(n_cities)
    visited_mask[0] = 1.0
    
    with torch.no_grad():
        for step in range(1, n_cities):
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
            next_city = unvisited_indices[next_idx]
            
            tour.append(next_city)
            visited_mask[next_city] = 1.0
    
    return np.array(tour)

def evaluate():
    device = torch.device('cpu')
    n_cities = 20
    
    # Load model
    model = TSPPolicy().to(device)
    model.load_state_dict(torch.load('models/tsp_policy.pt', map_location=device))
    print("Model loaded from models/tsp_policy.pt")
    
    # Load eval data
    eval_dir = 'data/eval_instances'
    eval_instances = []
    for filename in os.listdir(eval_dir):
        if filename.endswith('.json'):
            with open(os.path.join(eval_dir, filename), 'r') as f:
                data = json.load(f)
                eval_instances.append(data)
    
    print("Evaluating on %d instances" % len(eval_instances))
    
    nn_lengths = []
    learned_lengths = []
    opt_lengths = []
    
    for data in tqdm(eval_instances, desc="Evaluating"):
        coords = np.array(data['coords'])
        dists = dist_matrix(coords)
        
        nn_lengths.append(data['tour_length_nn'])
        opt_lengths.append(data['tour_length_opt'])
        
        learned_tour = build_greedy_tour(model, device, coords, n_cities)
        learned_lengths.append(tour_length(learned_tour, dists))
    
    # Results (using .format() - no f-strings)
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print("NN baseline:     %8.4f +/- %6.4f" % (np.mean(nn_lengths), np.std(nn_lengths)))
    print("Learned policy:  %8.4f +/- %6.4f" % (np.mean(learned_lengths), np.std(learned_lengths)))
    print("2-opt oracle:    %8.4f +/- %6.4f" % (np.mean(opt_lengths), np.std(opt_lengths)))
    
    improvement = 100 * (np.mean(nn_lengths) - np.mean(learned_lengths)) / np.mean(nn_lengths)
    print("Learned beats NN by: %5.1f%%" % improvement)
    print("="*50)

if __name__ == "__main__":
    evaluate()