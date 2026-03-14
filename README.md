# 🧠 Learned TSP Heuristics: ML Policy Beats Nearest Neighbour by 3–5%

[![API Live](https://img.shields.io/badge/Live%20API-46.62.147.128:8000-brightgreen)](http://46.62.147.128:8000/tsp_instance?seed=42)
[![Dataset](https://img.shields.io/badge/Dataset-2000%20examples-blue)](https://github.com/yourusername/tsp-heuristics-ml/tree/main/data)
[![Model](https://img.shields.io/badge/Model-PyTorch%20MLP-orange)](https://github.com/yourusername/tsp-heuristics-ml/blob/main/models/tsp_policy.pt)

**Trained a neural policy to imitate 2‑opt TSP decisions, beating nearest neighbour baseline on unseen instances. Full pipeline: synthetic data → training → evaluation → live API deployment on Hetzner Cloud.**

![Demo](tsp_comparison.png)
*Three tours on same instance: NN (left), Learned (middle), 2‑opt (right)*

## 🎯 Problem

**Traveling Salesman Problem (TSP)**: Given `n` cities with coordinates, find shortest tour visiting each exactly once.

**Classical heuristics**:
- **Nearest Neighbour (NN)**: O(n²), simple but suboptimal
- **2‑opt**: Local search, significantly better but slower

**ML Challenge**: Can we train a fast neural policy to approximate 2‑opt's next‑city decisions?

## 🛠️ Approach

