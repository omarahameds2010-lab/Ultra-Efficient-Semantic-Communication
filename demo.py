import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SemanticCommunicationModel
import json

def load_model(checkpoint_path='best_model.pth'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint['config']
    model = SemanticCommunicationModel(
        input_dim=config['input_dim'],
        latent_dim=config['latent_dim'],
        num_embeddings=config['num_embeddings']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def generate_test_signal(signal_type='sine', length=4096):
    """Generate different test signals"""
    if signal_type == 'sine':
        t = np.linspace(0, 4*np.pi, length)
        signal = np.sin(t * 3) + 0.5 * np.sin(t * 7)
    elif signal_type == 'square':
        signal = np.sign(np.sin(np.linspace(0, 8*np.pi, length)))
    elif signal_type == 'pulse':
        signal = np.zeros(length)
        pulse_positions = np.random.choice(length, size=50)
        signal[pulse_positions] = np.random.randn(50) * 3
    elif signal_type == 'noise':
        signal = np.random.randn(length)
    else:
        signal = np.random.randn(length)
    
    return torch.FloatTensor(signal).unsqueeze(0)

def calculate_metrics(original, reconstructed):
    """Calculate reconstruction metrics"""
    mse = torch.mean((original - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original - reconstructed)).item()
    
    # Signal-to-Noise Ratio
    signal_power = torch.mean(original ** 2).item()
    noise_power = torch.mean((original - reconstructed) ** 2).item()
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'SNR_dB': snr
    }

def visualize_compression(original, reconstructed, indices, signal_type):
    """Visualize the compression process"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Original Signal
    axes[0].plot(original.cpu().numpy()[0], color='#2E86AB', linewidth=1.5, label='Original Signal')
    axes[0].set_title(f'Original IoT Signal ({signal_type.upper()})', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    
    # Compressed Representation (Discrete Indices)
    axes[1].scatter(range(len(indices[0])), indices[0].cpu().numpy(), 
                   color='#A23B72', s=50, alpha=0.7, label='Codebook Indices')
    axes[1].set_title('Compressed Representation (Discrete Indices)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Index', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Reconstructed Signal
    axes[2].plot(original.cpu().numpy()[0], color='#2E86AB', linewidth=1.5, alpha=0.5, label='Original')
    axes[2].plot(reconstructed.cpu().numpy()[0], color='#F18F01', linewidth=1.5, label='Reconstructed')
    axes[2].set_title('Reconstructed Signal', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Sample Index', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'compression_demo_{signal_type}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: compression_demo_{signal_type}.png")
    plt.close()

def demo():
    """Run interactive demo"""
    print("="*70)
    print("🚀 SEMANTIC COMMUNICATION AI - LIVE DEMO")
    print("="*70)
    
    # Load model
    print("\n📥 Loading trained model...")
    model, checkpoint = load_model()
    
    config = checkpoint['config']
    compression_ratio = checkpoint['compression_ratio']
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Compression Ratio: {compression_ratio:.2f}%")
    print(f"📐 Architecture: {config['input_dim']} → {config['latent_dim']} → {config['num_embeddings']} embeddings")
    
    # Test different signal types
    signal_types = ['sine', 'square', 'pulse', 'noise']
    
    results = {}
    
    for signal_type in signal_types:
        print(f"\n{'='*70}")
        print(f"🔍 Testing {signal_type.upper()} Signal")
        print(f"{'='*70}")
        
        # Generate test signal
        test_signal = generate_test_signal(signal_type, length=config['input_dim'])
        
        # Compress and reconstruct
        with torch.no_grad():
            reconstructed, indices, z, quantized = model(test_signal)
        
        # Calculate metrics
        metrics = calculate_metrics(test_signal, reconstructed)
        
        print(f"📈 Reconstruction Metrics:")
        print(f"   MSE: {metrics['MSE']:.6f}")
        print(f"   MAE: {metrics['MAE']:.6f}")
        print(f"   SNR: {metrics['SNR_dB']:.2f} dB")
        
        # Original size vs Compressed size
        original_bits = config['input_dim'] * 32  # float32
        compressed_bits = np.log2(config['num_embeddings'])
        
        print(f"\n💾 Data Size:")
        print(f"   Original: {original_bits} bits ({original_bits/8:.0f} bytes)")
        print(f"   Compressed: {compressed_bits:.1f} bits ({compressed_bits/8:.2f} bytes)")
        print(f"   Compression: {compression_ratio:.2f}%")
        
        # Visualize
        visualize_compression(test_signal, reconstructed, indices, signal_type)
        
        results[signal_type] = metrics
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY ACROSS ALL SIGNAL TYPES")
    print(f"{'='*70}")
    
    avg_mse = np.mean([results[st]['MSE'] for st in signal_types])
    avg_snr = np.mean([results[st]['SNR_dB'] for st in signal_types])
    
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average SNR: {avg_snr:.2f} dB")
    print(f"Compression Ratio: {compression_ratio:.2f}%")
    
    print(f"\n{'='*70}")
    print("✅ Demo Complete! Check the generated PNG files.")
    print(f"{'='*70}")

if __name__ == "__main__":
    demo()
