#!/bin/bash

echo "=========================================="
echo "🚀 Semantic Communication AI - Quick Start"
echo "=========================================="

# Check Python version
echo ""
echo "📋 Checking Python version..."
python --version

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

# Run a quick test
echo ""
echo "🧪 Running quick model test..."
python -c "
from model import SemanticCommunicationModel, calculate_compression_ratio
import torch

model = SemanticCommunicationModel(input_dim=4096, latent_dim=2, num_embeddings=512)
comp_ratio, orig, comp = calculate_compression_ratio(4096, 2, 512)

print(f'\n✅ Model initialized successfully!')
print(f'📊 Compression Ratio: {comp_ratio:.2f}%')
print(f'💾 Original: {orig} bits → Compressed: {comp:.1f} bits')

# Test inference
test_input = torch.randn(4, 4096)
output, indices, z, q = model(test_input)
print(f'\n🎯 Test inference passed!')
print(f'   Input shape: {test_input.shape}')
print(f'   Output shape: {output.shape}')
print(f'   Indices transmitted: {indices.tolist()}')
"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Train the model:    python train.py"
echo "  2. Run the demo:       python demo.py"
echo ""
echo "For full documentation, see README.md"
echo "=========================================="
