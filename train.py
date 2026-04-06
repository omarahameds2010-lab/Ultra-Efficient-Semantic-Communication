import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from tqdm import tqdm
import json

class SemanticCommunicationModel(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=64, num_embeddings=512):
        super(SemanticCommunicationModel, self).__init__()
        
        # 1. Deep Semantic Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, latent_dim)
        )
        
        # 2. Vector Quantization Layer (The Shared Codebook)
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # 3. Semantic Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, input_dim)
        )
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
    
    def forward(self, x):
        # Step 1: Extract Semantic Features
        z = self.encoder(x)
        
        # Step 2: Vector Quantization (Finding the nearest neighbor)
        distances = (torch.sum(z**2, dim=1, keepdim=True) 
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z, self.embedding.weight.t()))
        
        # Get the discrete indices
        indices = torch.argmin(distances, dim=1)
        quantized_vectors = self.embedding(indices)
        
        # Step 3: Straight-Through Estimator (STE)
        quantized = z + (quantized_vectors - z).detach()
        
        # Step 4: Reconstruction
        output = self.decoder(quantized)
        
        return output, indices, z, quantized_vectors

def calculate_loss(outputs, inputs, z, quantized, beta=0.25):
    """Calculate combined VQ-VAE loss"""
    recon_loss = F.mse_loss(outputs, inputs)
    vq_loss = F.mse_loss(quantized.detach(), z)
    commit_loss = F.mse_loss(quantized, z.detach())
    total_loss = recon_loss + vq_loss + beta * commit_loss
    
    return total_loss, recon_loss, vq_loss, commit_loss

def calculate_compression_ratio(input_dim, latent_dim, num_embeddings, batch_size=1):
    """Calculate actual compression ratio"""
    # Original data: batch_size * input_dim * 32 bits (float32)
    original_bits = batch_size * input_dim * 32
    
    # Compressed data: batch_size * log2(num_embeddings) bits (just indices)
    compressed_bits = batch_size * np.log2(num_embeddings)
    
    compression_ratio = (1 - compressed_bits / original_bits) * 100
    return compression_ratio

def generate_synthetic_iot_data(num_samples=10000, input_dim=1024, noise_level=0.1):
    """Generate synthetic IoT sensor data with patterns"""
    data = []
    for _ in range(num_samples):
        # Simulate different sensor patterns
        pattern_type = np.random.choice(['sine', 'square', 'noise', 'pulse'])
        
        if pattern_type == 'sine':
            t = np.linspace(0, 4*np.pi, input_dim)
            signal = np.sin(t * np.random.uniform(1, 5))
        elif pattern_type == 'square':
            signal = np.random.choice([-1, 1], size=input_dim)
        elif pattern_type == 'pulse':
            signal = np.zeros(input_dim)
            pulse_positions = np.random.choice(input_dim, size=int(input_dim*0.1))
            signal[pulse_positions] = np.random.randn(len(pulse_positions))
        else:
            signal = np.random.randn(input_dim)
        
        # Add noise
        signal += np.random.randn(input_dim) * noise_level
        data.append(signal)
    
    return torch.FloatTensor(np.array(data))

def train_epoch(model, dataloader, optimizer, device, beta=0.25):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_vq = 0
    total_commit = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data,) in enumerate(pbar):
        data = data.to(device)
        
        optimizer.zero_grad()
        outputs, indices, z, quantized = model(data)
        loss, recon, vq, commit = calculate_loss(outputs, data, z, quantized, beta)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon.item()
        total_vq += vq.item()
        total_commit += commit.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon.item():.4f}'
        })
    
    n = len(dataloader)
    return total_loss/n, total_recon/n, total_vq/n, total_commit/n

def validate(model, dataloader, device, beta=0.25):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_recon = 0
    
    with torch.no_grad():
        for data, in dataloader:
            data = data.to(device)
            outputs, indices, z, quantized = model(data)
            loss, recon, vq, commit = calculate_loss(outputs, data, z, quantized, beta)
            
            total_loss += loss.item()
            total_recon += recon.item()
    
    n = len(dataloader)
    return total_loss/n, total_recon/n

def main():
    # Hyperparameters
    INPUT_DIM = 4096  # Massive IoT scenario
    LATENT_DIM = 2  # Ultra-compressed representation
    NUM_EMBEDDINGS = 512
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    BETA = 0.25
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Training on: {device}")
    
    # Calculate compression ratio
    compression_ratio = calculate_compression_ratio(INPUT_DIM, LATENT_DIM, NUM_EMBEDDINGS)
    print(f"📊 Theoretical Compression Ratio: {compression_ratio:.2f}%")
    
    # Generate data
    print("📡 Generating synthetic IoT data...")
    train_data = generate_synthetic_iot_data(num_samples=8000, input_dim=INPUT_DIM)
    val_data = generate_synthetic_iot_data(num_samples=2000, input_dim=INPUT_DIM)
    
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = SemanticCommunicationModel(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        num_embeddings=NUM_EMBEDDINGS
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'val_recon': []
    }
    
    print(f"\n🔥 Starting Training...")
    print(f"📐 Architecture: {INPUT_DIM} → {LATENT_DIM} → {NUM_EMBEDDINGS} embeddings")
    print("="*70)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_recon, train_vq, train_commit = train_epoch(
            model, train_loader, optimizer, device, BETA
        )
        
        # Validate
        val_loss, val_recon = validate(model, val_loader, device, BETA)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['val_recon'].append(val_recon)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Recon: {train_recon:.4f} | VQ: {train_vq:.4f} | Commit: {train_commit:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'compression_ratio': compression_ratio,
                'config': {
                    'input_dim': INPUT_DIM,
                    'latent_dim': LATENT_DIM,
                    'num_embeddings': NUM_EMBEDDINGS
                }
            }, 'best_model.pth')
            print(f"✅ Best model saved! (Val Loss: {val_loss:.4f})")
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print(f"🎉 Training Complete!")
    print(f"📊 Best Validation Loss: {best_val_loss:.4f}")
    print(f"🗜️ Compression Ratio: {compression_ratio:.2f}%")
    print(f"💾 Model saved to: best_model.pth")
    print("="*70)

if __name__ == "__main__":
    main()
