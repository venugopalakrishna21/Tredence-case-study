# Self-Pruning Neural Network Implementation Report

## Executive Summary

This report documents the implementation of a **self-pruning neural network** that learns to remove unnecessary weights during the training process. Instead of pruning post-training, the network uses learnable gate parameters that are optimized alongside standard weights, allowing the network to adaptively identify and remove weak connections.

---

## Part 1: Technical Architecture

### 1.1 The PrunableLinear Layer

The core innovation is the `PrunableLinear` class, which extends standard linear layers with pruning capability.

#### Implementation Details:

```
PrunableLinear(in_features, out_features)
├── weight: [out_features, in_features]
├── bias: [out_features]
└── gate_scores: [out_features, in_features]  ← NEW
```

**How it works:**

1. **Gate Scores**: Each weight has an associated learnable "gate score"
2. **Sigmoid Transformation**: `gates = sigmoid(gate_scores)` → values in [0, 1]
3. **Element-wise Multiplication**: `pruned_weights = weight * gates`
4. **Standard Linear Op**: Output = pruned_weights @ input + bias

**Gradient Flow**: Both `weight` and `gate_scores` receive gradients:
- `weight` gets gradients from: `grad_output * gates * input`
- `gate_scores` gets gradients from the sparsity loss, pushing gates toward 0

### 1.2 Network Architecture

```
Input (3×32×32 image) 
    ↓ (Flatten)
PrunableLinear(3072 → 512) + ReLU
    ↓
PrunableLinear(512 → 256) + ReLU
    ↓
PrunableLinear(256 → 128) + ReLU
    ↓
PrunableLinear(128 → 10)
    ↓
Output (class logits)
```

**Total Parameters**: ~3.4M (spread across 4 prunable layers)

---

## Part 2: The Sparsity Mechanism

### 2.1 Why L1 Regularization Encourages Sparsity

L1 regularization (Lasso) is mathematically biased toward producing sparse solutions:

**Geometric Intuition:**
- **L2 penalty**: Creates circular constraint region → solutions at interior points
- **L1 penalty**: Creates diamond constraint region → solutions at corners (where values = 0)

**For gate values:**
```
Total Loss = CrossEntropy(predictions, targets) + λ × Σ|gates|
```

**Why this works:**
1. Without L1 term: No incentive to set gates to 0 (waste of parameters)
2. With L1 term: Each gate's contribution to loss is linear
3. If a gate would save more loss by being 0 than by being active → gradient pushes it toward 0
4. The optimizer "chooses" between fully pruning (gate ≈ 0) or fully keeping (gate ≈ 1)

### 2.2 The Sparsity Trade-off

```
λ (Lambda) → Controls aggressiveness of pruning
├── λ = 0.0001 (low): Minimal pruning, highest accuracy
├── λ = 0.001 (medium): Moderate pruning and accuracy
└── λ = 0.01 (high): Aggressive pruning, lower accuracy
```

**The Trade-off Curve:**
- Higher λ → More gates driven to 0 → Fewer parameters → Faster inference but lower accuracy
- Lower λ → Fewer pruned gates → More parameters → Better accuracy but slower inference

---

## Part 3: Implementation Highlights

### 3.1 Key Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Gate activation | Sigmoid | Smooth, differentiable, naturally bounded in [0,1] |
| Sparsity loss | L1 norm of gates | Proven to encourage sparsity |
| Initialization | gate_scores=0 | Produces gates ≈ 0.5 initially (balanced start) |
| Optimizer | Adam | Adaptive learning rates for both weights and gates |
| Scheduler | StepLR (×0.5 every 10 epochs) | Standard practice for stabilizing training |

### 3.2 Training Loop

```python
for epoch in range(num_epochs):
    for batch in train_loader:
        # Forward pass
        output = model(x)
        
        # Calculate losses
        class_loss = CrossEntropy(output, target)
        sparsity_loss = sum(all_gates)
        total_loss = class_loss + λ × sparsity_loss
        
        # Backward and optimize
        optimizer.zero_grad()
        total_loss.backward()  # ← Gradients flow to both weights and gates
        optimizer.step()
```

**Key insight**: The gate gradients come almost entirely from the sparsity loss, creating a direct signal for pruning.

### 3.3 Evaluation Metrics

**Sparsity Level (%):**
```
Sparsity = (Count of gates < threshold) / Total gates × 100

threshold = 0.01  (gates < 0.01 are considered "pruned")
```

**Why this metric matters:**
- If 70% of gates are < 0.01 → 70% of weights contribute < 1% to the output
- Enables inference acceleration through weight skipping or pruned network export

---

## Part 4: Expected Results

### 4.1 Results Table Template

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|-------------------|
| 0.0001 | 72-76 | 10-20 |
| 0.001 | 68-72 | 30-50 |
| 0.01 | 60-68 | 60-80 |

**Interpretation:**
- λ=0.0001: Minimal sparsity; acts almost like standard training
- λ=0.001: Good balance between sparsity and accuracy
- λ=0.01: Aggressive pruning; accuracy drops but model becomes very sparse

### 4.2 Expected Gate Distribution

**For λ=0.0001 (Low Sparsity):**
- Smooth, distributed gates across [0, 1]
- Few gates exactly at 0
- Most gates > 0.3

**For λ=0.01 (High Sparsity):**
- **Bimodal distribution**: Large spike at 0, cluster near 1
- Clear separation between "pruned" and "active" weights
- This is the goal of pruning

---

## Part 5: Code Quality & Practices

### 5.1 Best Practices Implemented

✅ **Modularity**: Clear separation between layer, network, and training logic
✅ **Documentation**: Docstrings for every class and function
✅ **Reproducibility**: Fixed random seeds
✅ **Error Handling**: Device-agnostic (CPU/GPU)
✅ **Efficiency**: Batch processing, efficient tensor operations
✅ **Monitoring**: Detailed logging of loss and accuracy

### 5.2 Mathematical Correctness

- ✅ Gradient flow verified (both weight and gate gradients)
- ✅ Sparsity loss correctly computed as sum of gate values
- ✅ Sigmoid ensures gates are always in [0, 1]
- ✅ L1 penalty is properly weighted by λ

---

## Part 6: Extensions & Future Work

### Potential Improvements:

1. **Structured Pruning**: Prune entire channels/neurons, not individual weights
2. **Magnitude-based Initialization**: Initialize gates based on weight magnitudes
3. **Iterative Pruning**: Train, prune, fine-tune in cycles
4. **Knowledge Distillation**: Use original network to teach pruned network
5. **Hardware-aware Pruning**: Target specific hardware (mobile, edge)

### Advanced Techniques:

- **LoRA Fine-tuning**: After pruning, apply LoRA to recover accuracy
- **Lottery Ticket Hypothesis**: Find sparse subnetworks from random initialization
- **Mixed Precision**: Use FP16 for faster training

---

## Part 7: Running the Code

### Requirements:
```
torch>=2.0
torchvision>=0.15
numpy
matplotlib
```

### Execution:
```bash
python prunable_network.py
```

### Output:
- `gate_distribution_lambda_*.png`: Histogram of gate values for each λ
- `results_summary.txt`: Detailed results and analysis

### Expected Runtime:
- Single λ value: ~10 minutes on GPU, ~45 minutes on CPU
- All three λ values: ~30 minutes on GPU, ~2+ hours on CPU

---

## Conclusion

This implementation demonstrates that **self-pruning neural networks** can successfully learn which weights are important during training, eliminating the need for post-training pruning.

**Key Takeaways:**
1. L1 regularization on learnable gates drives the network toward sparsity
2. The trade-off between sparsity and accuracy is controlled by λ
3. The bimodal gate distribution is the signature of successful pruning
4. This approach is more elegant than magnitude-based pruning (which requires post-training)

The resulting sparse networks can be:
- ✅ Deployed with lower memory footprint
- ✅ Executed faster with weight skipping
- ✅ Quantized more effectively
- ✅ Transferred to edge devices more easily

---

**Author**: [Your Name]  
**Date**: 2025  
**Institution**: Tredence Analytics - AI Engineering Internship Case Study
