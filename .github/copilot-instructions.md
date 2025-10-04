# AI Coding Agent Instructions for SparseLoRA

## Project Overview
This is a research project developing **sparse LoRA (Low-Rank Adaptation)** for large language models, focused on creating interpretable adapters where latents aim to be **monosemantic** (one concept per latent). The core innovation is using **TopK sparsity** in an expanded latent space to improve interpretability.

## Architecture & Key Components

### Core Modules
- **`src/sft.py`**: Enhanced SFT training with `TopKLoRALinearSTE` modules and regularization (decorrelation, orthogonality, mass constraints)
- **`src/models.py`**: Custom TopK LoRA implementations (`TopKLoRALinear`, `TopKLoRALinearSTE`) with STE (Straight-Through Estimator)
- **`src/dpo.py`**: DPO training with TopK-aware regularizers for sparse latent learning
- **`src/evals.py`**: Comprehensive evaluation including toxicity, instruction following, and auto-interpretation
- **`src/autointerp.py`**: Automated interpretation of sparse latents using external LLMs
- **`src/cosine_test.py`**: Cosine similarity analysis for understanding latent semantics

### Training Pipeline
1. **SFT Phase**: Train sparse LoRA adapters with TopK constraints
2. **Optional DPO**: Further alignment training with preserved sparsity
3. **Evaluation**: Multi-faceted assessment including interpretability metrics

## Development Patterns

### Configuration System (Hydra)
Uses hierarchical configs in `config/train_config/` and `config/eval_config/`:
```yaml
# Example override pattern
python main.py training=sft_enhanced_sparse training.model=gemma_2b
```
- **Default configs**: `config/train_config/default.yaml`, `config/eval_config/default.yaml`
- **Modular overrides**: `training/`, `model/`, `dataset/`, `experiment/` subdirectories
- **Key pattern**: Each config file uses `defaults:` list to compose from other configs

### TopK LoRA Implementation
The core innovation uses `TopKLoRALinearSTE` with these key parameters:
- **`r`**: Total latent space size (e.g., 512, 1024)
- **`k`**: Active latents via TopK selection (e.g., 4, 8, 64)
- **`k_schedule`**: Dynamic sparsification during training ("linear", "cubic")
- **`temperature`**: Gumbel-style soft TopK annealing

### Regularization Framework
Training includes multiple regularizers for interpretable latents:
- **L_DECORR**: Decorrelation between latents
- **L_MASS**: Mass concentration (encourage sparsity)
- **L_ORTHO_A/B**: Orthogonality constraints on LoRA matrices
- **Schedule**: Most regularizers use cubic scheduling `t³` for gradual onset

## Key Workflows

### Training
```bash
# Basic SFT with sparse LoRA
python main.py training=sft_enhanced_sparse

# With model override
python main.py training=sft_enhanced_sparse training.model=gemma_2b

# Full pipeline (SFT + DPO)
python main.py training=all training.model=gemma_2b
```

### Evaluation
```bash
# Run default eval suite
python eval.py

# Specific evaluations
python eval.py 'defaults=[_self_, logger: wandb_disabled, evals/toxicity@evals.toxicity, evals/auto-interpret@evals.auto_interp]'
```

### Environment Setup
- Requires `.env` file from `.env-template` for API keys (toxicity eval, auto-interpretation)
- Uses conda/pip environment from `requirements.txt`
- CUDA-enabled for training (A40/V100 class GPUs recommended)

## File Conventions

### Model Storage
- **Adapters**: `adapters/sft/` and `adapters/sparsesft/` for trained models
- **Cache**: `cache/` for activation caches during interpretation
- **Outputs**: `outputs/YYYY-MM-DD/` for timestamped results

### Experiment Tracking
- **WandB**: Default logging (can disable with `logger: wandb_disabled`)
- **Logs**: Training logs in timestamped `outputs/` directories
- **Checkpoints**: Regular saving every 500 steps by default

## Critical Implementation Details

### TopK Sparsity
- Uses **Straight-Through Estimator** for gradient flow through discrete TopK
- **Energy normalization**: Scale by `α/k` to maintain signal strength
- **Dynamic k**: Linear/cubic schedules reduce active latents during training
- **Layer targeting**: Often focus on specific layers (e.g., `layers.13.*`) for analysis

### Data Handling
- **Chat format**: Uses `trl.setup_chat_format()` for conversation datasets
- **Completion-only loss**: `completion_only_loss: True` focuses training on assistant responses
- **Mixed datasets**: Support for combining SFT datasets via config composition

### Memory Management
- **Gradient checkpointing**: Essential for large models (`gradient_checkpointing: True`)
- **4-bit quantization**: Available via `quantization: 4bit` config
- **Cache clearing**: Custom callbacks clear TopK internal caches to prevent memory leaks

## Integration Points

### External APIs
- **Perspective API**: For toxicity evaluation (requires API key in `.env`)
- **OpenAI/Anthropic**: For auto-interpretation of latents
- **WandB**: For experiment tracking and hyperparameter logging

### Model Loading
- **HuggingFace Integration**: All models loaded via `transformers.AutoModelForCausalLM`
- **PEFT Integration**: LoRA adapters managed through `peft` library
- **Custom wrappers**: `TopKLoRALinearSTE` replaces standard LoRA layers during training

When modifying this codebase:
1. **Preserve sparsity**: Maintain TopK constraints in custom modules
2. **Config consistency**: Use Hydra patterns for new configurations
3. **Regularization**: Consider impact on interpretability objectives
4. **Memory**: Be mindful of CUDA memory with large latent spaces
5. **Evaluation**: Include interpretability metrics alongside standard benchmarks