import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
from tqdm import tqdm

class TSPLabelDataset(Dataset):
    def __init__(self, data_dir):
        self.examples = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    self.examples.extend(data['examples'])
        print(f"✅ Loaded {len(self.examples)} examples from {data_dir}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        state = torch.FloatTensor(ex['state'])  # shape: (97,)
        target = torch.LongTensor([ex['next']]) # shape: (1,)
        return state, target

class TSPPolicy(nn.Module):
    def __init__(self, input_dim=97, n_cities=20, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_cities)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # logits over 20 cities

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Data (your 100 instances = 2000 examples)
    train_dataset = TSPLabelDataset('data/train_instances')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Model + optimizer
    model = TSPPolicy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(50):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/50')
        for batch_idx, (states, targets) in enumerate(pbar):
            states, targets = states.to(device), targets.to(device).squeeze()
            
            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})
        
        avg_loss = total_loss / len(train_loader)
        final_acc = 100. * correct / total
        print(f"📊 Epoch {epoch+1}: Loss={avg_loss:.4f}, Final Acc={final_acc:.2f}%")
    
    # Save trained model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/tsp_policy.pt')
    print("✅ Model saved to models/tsp_policy.pt")
    print("🎉 Training complete! Next: python evaluate.py")

if __name__ == "__main__":
    train_model()
