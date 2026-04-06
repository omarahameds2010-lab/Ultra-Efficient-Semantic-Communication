"""
Semantic Communication Model Architecture
==========================================

This module implements a Vector-Quantized Variational Autoencoder (VQ-VAE)
for ultra-efficient semantic communication in IoT systems.

Author: Omar
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SemanticCommunicationModel(nn.Module):
    """
    Semantic Communication Model with Vector Quantization
    
    Architecture:
        Input → Deep Encoder → VQ Bottleneck → Decoder → Reconstructed Output
        
    Key Components:
        1. Deep Semantic Encoder: Extracts semantic features
        2. Vector Quantization: Discretizes latent space using codebook
        3. Semantic Decoder: Reconstructs original signal
        
    Args:
        input_dim (int): Dimension of input signal (default: 1024)
        latent_dim (int): Dimension of latent semantic space (default: 64)
        num_embeddings (int): Size of VQ codebook (default: 512)
    """
    
    def __init__(self, input_dim=1024, latent_dim=64, num_embeddings=512):
        super(SemanticCommunicationModel, self).__init__()
        
        # Model configuration
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        
        # 1. Deep Semantic Encoder
        # Progressively reduces dimensionality while extracting semantic features
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
        # This is the discrete bottleneck that enables massive compression
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
        # 3. Semantic Decoder
        # Mirrors the encoder to reconstruct the original signal
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        """
        Forward pass through the semantic communication model
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]
            
        Returns:
            output (torch.Tensor): Reconstructed signal [batch_size, input_dim]
            indices (torch.Tensor): Discrete codebook indices [batch_size]
            z (torch.Tensor): Continuous latent features [batch_size, latent_dim]
            quantized_vectors (torch.Tensor): Quantized features [batch_size, latent_dim]
        """
        
        # Step 1: Extract Semantic Features
        # Input: [batch_size, input_dim] → Output: [batch_size, latent_dim]
        z = self.encoder(x)
        
        # Step 2: Vector Quantization (Finding the nearest neighbor)
        # Calculate L2 distances between encoder output and all codebook vectors
        # Formula: ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        distances = (
            torch.sum(z**2, dim=1, keepdim=True) 
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )
        
        # Get the discrete indices of nearest codebook vectors
        # This is what gets transmitted (massive compression!)
        indices = torch.argmin(distances, dim=1)
        quantized_vectors = self.embedding(indices)
        
        # Step 3: Straight-Through Estimator (STE)
        # Trick: Forward uses quantized, backward uses continuous z
        # This allows gradients to flow through the discrete operation
        quantized = z + (quantized_vectors - z).detach()
        
        # Step 4: Reconstruction
        # Decode the quantized features back to original space
        output = self.decoder(quantized)
        
        return output, indices, z, quantized_vectors
    
    def encode(self, x):
        """
        Encode input to discrete indices (for transmission)
        
        Args:
            x (torch.Tensor): Input signal [batch_size, input_dim]
            
        Returns:
            indices (torch.Tensor): Discrete codebook indices [batch_size]
        """
        with torch.no_grad():
            z = self.encoder(x)
            distances = (
                torch.sum(z**2, dim=1, keepdim=True) 
                + torch.sum(self.embedding.weight**2, dim=1)
                - 2 * torch.matmul(z, self.embedding.weight.t())
            )
            indices = torch.argmin(distances, dim=1)
        return indices
    
    def decode(self, indices):
        """
        Decode discrete indices to reconstructed signal (at receiver)
        
        Args:
            indices (torch.Tensor): Codebook indices [batch_size]
            
        Returns:
            output (torch.Tensor): Reconstructed signal [batch_size, input_dim]
        """
        with torch.no_grad():
            quantized_vectors = self.embedding(indices)
            output = self.decoder(quantized_vectors)
        return output
    
    def get_codebook_usage(self, dataloader, device='cpu'):
        """
        Analyze which codebook entries are being used
        
        Args:
            dataloader (DataLoader): Data to analyze
            device (str): Device to run on
            
        Returns:
            usage_stats (dict): Statistics about codebook usage
        """
        self.eval()
        self.to(device)
        
        usage_count = torch.zeros(self.num_embeddings)
        
        with torch.no_grad():
            for batch, in dataloader:
                batch = batch.to(device)
                _, indices, _, _ = self.forward(batch)
                for idx in indices:
                    usage_count[idx.item()] += 1
        
        used_codes = (usage_count > 0).sum().item()
        total_assignments = usage_count.sum().item()
        
        return {
            'total_codes': self.num_embeddings,
            'used_codes': used_codes,
            'usage_percentage': (used_codes / self.num_embeddings) * 100,
            'total_assignments': int(total_assignments),
            'usage_distribution': usage_count.numpy()
        }


def calculate_loss(outputs, inputs, z, quantized, beta=0.25):
    """
    Calculate VQ-VAE loss with three components
    
    Args:
        outputs (torch.Tensor): Reconstructed signals
        inputs (torch.Tensor): Original signals
        z (torch.Tensor): Continuous latent features
        quantized (torch.Tensor): Quantized latent features
        beta (float): Weight for commitment loss
        
    Returns:
        total_loss (torch.Tensor): Combined loss
        recon_loss (torch.Tensor): Reconstruction loss (MSE)
        vq_loss (torch.Tensor): Codebook update loss
        commit_loss (torch.Tensor): Encoder commitment loss
    """
    
    # 1. Reconstruction Loss (MSE)
    # Measures how well we can reconstruct the original signal
    recon_loss = F.mse_loss(outputs, inputs)
    
    # 2. VQ Loss: Update codebook to match encoder outputs
    # Moves codebook vectors closer to encoder outputs
    vq_loss = F.mse_loss(quantized.detach(), z)
    
    # 3. Commitment Loss: Force encoder to commit to codebook
    # Prevents encoder from growing arbitrarily large
    commit_loss = F.mse_loss(quantized, z.detach())
    
    # Total Combined Loss
    total_loss = recon_loss + vq_loss + beta * commit_loss
    
    return total_loss, recon_loss, vq_loss, commit_loss


def calculate_compression_ratio(input_dim, latent_dim, num_embeddings, batch_size=1):
    """
    Calculate theoretical compression ratio
    
    Args:
        input_dim (int): Input signal dimension
        latent_dim (int): Latent space dimension
        num_embeddings (int): Codebook size
        batch_size (int): Batch size for calculation
        
    Returns:
        compression_ratio (float): Compression percentage
        original_bits (int): Original data size in bits
        compressed_bits (float): Compressed data size in bits
    """
    
    # Original data: batch_size * input_dim * 32 bits (float32)
    original_bits = batch_size * input_dim * 32
    
    # Compressed data: batch_size * log2(num_embeddings) bits (just indices)
    compressed_bits = batch_size * np.log2(num_embeddings)
    
    # Compression ratio as percentage
    compression_ratio = (1 - compressed_bits / original_bits) * 100
    
    return compression_ratio, original_bits, compressed_bits


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = SemanticCommunicationModel(
        input_dim=4096,
        latent_dim=2,
        num_embeddings=512
    )
    
    # Calculate compression ratio
    comp_ratio, orig_bits, comp_bits = calculate_compression_ratio(
        input_dim=4096,
        latent_dim=2,
        num_embeddings=512
    )
    
    print(f"Model Architecture:")
    print(f"  Input: {model.input_dim} dimensions")
    print(f"  Latent: {model.latent_dim} dimensions")
    print(f"  Codebook: {model.num_embeddings} entries")
    print(f"\nCompression:")
    print(f"  Original: {orig_bits} bits")
    print(f"  Compressed: {comp_bits:.1f} bits")
    print(f"  Ratio: {comp_ratio:.2f}%")
    
    # Test forward pass
    batch = torch.randn(8, 4096)
    outputs, indices, z, quantized = model(batch)
    
    print(f"\nForward Pass:")
    print(f"  Input shape: {batch.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Transmitted indices: {indices[:5].tolist()}")
