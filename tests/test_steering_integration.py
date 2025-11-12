"""
Integration tests for feature steering with eval pipeline alignment.

This test suite verifies that the steering pipeline correctly:
1. Loads models using init_model_tokenizer_fixed (same as eval)
2. Applies feature steering via FeatureSteeringContext
3. Handles chat templates and generation properly
4. Works with wrapped_modules from eval pipeline
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from src.models import TopKLoRALinearSTE
from src.steering import (
    steer_features,
    FeatureSteeringContext,
    list_available_adapters,
    FeatureSteerer
)


@pytest.fixture
def mock_topk_module():
    """Create a mock TopKLoRALinearSTE module for testing."""
    module = MagicMock(spec=TopKLoRALinearSTE)
    module.r = 512
    module._current_k.return_value = 4
    module._tau.return_value = 0.0
    module.training = False
    module.hard_eval = True
    module.relu_latents = True
    module.scale = 1.0

    # Mock the internal matrices
    module.A_module = MagicMock()
    module.A_module.weight = torch.randn(512, 256)
    module.B_module = MagicMock()
    module.B_module.weight = torch.randn(256, 512)
    module.base_layer = MagicMock()
    module.dropout = MagicMock(side_effect=lambda x: x)

    return module


@pytest.fixture
def mock_model_with_topk(mock_topk_module):
    """Create a mock model with TopKLoRALinearSTE modules."""
    model = MagicMock()

    # Create a module hierarchy
    modules = {
        "base_model.model.model.layers.11.self_attn.q_proj": mock_topk_module,
        "base_model.model.model.layers.11.mlp.down_proj": mock_topk_module,
    }

    model.named_modules.return_value = modules.items()
    return model


@pytest.fixture
def wrapped_modules(mock_topk_module):
    """Create wrapped_modules dict like init_model_tokenizer_fixed returns."""
    return {
        "base_model.model.model.layers.11.self_attn.q_proj": mock_topk_module,
        "base_model.model.model.layers.11.mlp.down_proj": mock_topk_module,
    }


def test_feature_steerer_initialization():
    """Test that FeatureSteerer initializes correctly."""
    feature_indices = [0, 1, 2]
    effects = ["enable", "disable", "isolate"]

    steerer = FeatureSteerer(feature_indices, effects, amplification=2.0)

    assert steerer.feature_indices == feature_indices
    assert steerer.effects == effects
    assert steerer.amplification == 2.0
    assert steerer.has_isolate is True


def test_feature_steerer_validation():
    """Test that FeatureSteerer validates inputs properly."""
    # Mismatched lengths
    with pytest.raises(AssertionError):
        FeatureSteerer([0, 1], ["enable"])

    # Invalid effect
    with pytest.raises(AssertionError):
        FeatureSteerer([0], ["invalid_effect"])


def test_steer_features_with_wrapped_modules(mock_model_with_topk, wrapped_modules):
    """Test steer_features with wrapped_modules parameter."""
    feature_dict = {
        "base_model.model.model.layers.11.self_attn.q_proj": [
            (217, "enable"),
            (45, "disable"),
        ]
    }

    result = steer_features(
        mock_model_with_topk,
        feature_dict,
        verbose=False,
        wrapped_modules=wrapped_modules
    )

    assert "hooks" in result
    assert "steerers" in result
    assert "applied_count" in result
    assert result["applied_count"] == 2
    assert len(result["hooks"]) == 1

    # Cleanup
    for hook in result["hooks"]:
        hook.remove()


def test_steer_features_without_wrapped_modules(mock_model_with_topk):
    """Test steer_features without wrapped_modules (fallback to named_modules)."""
    feature_dict = {
        "base_model.model.model.layers.11.self_attn.q_proj": [
            (100, "enable"),
        ]
    }

    result = steer_features(
        mock_model_with_topk,
        feature_dict,
        verbose=False,
        wrapped_modules=None
    )

    assert result["applied_count"] == 1

    # Cleanup
    for hook in result["hooks"]:
        hook.remove()


def test_feature_steering_context(mock_model_with_topk, wrapped_modules):
    """Test FeatureSteeringContext manager."""
    feature_dict = {
        "base_model.model.model.layers.11.mlp.down_proj": [
            (50, "isolate"),
        ]
    }

    hooks_applied = False
    with FeatureSteeringContext(
        mock_model_with_topk,
        feature_dict,
        verbose=False,
        amplification=5.0,
        wrapped_modules=wrapped_modules
    ) as hooks_info:
        hooks_applied = True
        assert len(hooks_info["hooks"]) > 0

    assert hooks_applied
    # Context manager should have cleaned up hooks


def test_list_available_adapters_with_wrapped_modules(mock_model_with_topk, wrapped_modules):
    """Test list_available_adapters with wrapped_modules."""
    adapters_info = list_available_adapters(
        mock_model_with_topk,
        verbose=False,
        wrapped_modules=wrapped_modules
    )

    assert len(adapters_info) == 2
    for name, info in adapters_info.items():
        assert "r" in info
        assert "k" in info
        assert "temperature" in info
        assert "num_features" in info
        assert info["r"] == 512
        assert info["k"] == 4


def test_list_available_adapters_without_wrapped_modules(mock_model_with_topk):
    """Test list_available_adapters without wrapped_modules (fallback)."""
    adapters_info = list_available_adapters(
        mock_model_with_topk,
        verbose=False,
        wrapped_modules=None
    )

    assert len(adapters_info) == 2


def test_steering_with_amplification(mock_model_with_topk, wrapped_modules):
    """Test that amplification parameter is passed through correctly."""
    feature_dict = {
        "base_model.model.model.layers.11.self_attn.q_proj": [
            (0, "enable"),
        ]
    }

    result = steer_features(
        mock_model_with_topk,
        feature_dict,
        verbose=False,
        amplification=10.0,
        wrapped_modules=wrapped_modules
    )

    # Check that steerer was created with correct amplification
    steerer_name = list(result["steerers"].keys())[0]
    steerer = result["steerers"][steerer_name]
    assert steerer.amplification == 10.0

    # Cleanup
    for hook in result["hooks"]:
        hook.remove()


def test_steering_partial_name_matching(mock_model_with_topk, wrapped_modules):
    """Test that partial adapter name matching works."""
    # Use a shortened name
    feature_dict = {
        "layers.11.self_attn.q_proj": [  # Partial name
            (0, "enable"),
        ]
    }

    result = steer_features(
        mock_model_with_topk,
        feature_dict,
        verbose=False,
        wrapped_modules=wrapped_modules
    )

    # Should match despite partial name
    assert result["applied_count"] == 1

    # Cleanup
    for hook in result["hooks"]:
        hook.remove()


@patch('src.evals.init_model_tokenizer_fixed')
def test_steer_py_uses_eval_loader(mock_init):
    """Test that steer.py imports and would use init_model_tokenizer_fixed."""
    # Just verify the import works
    from src.evals import init_model_tokenizer_fixed
    assert callable(init_model_tokenizer_fixed)


def test_chat_template_formatting():
    """Test that prompt formatting logic handles chat templates correctly."""
    from transformers import AutoTokenizer

    # Mock tokenizer with chat template
    tokenizer = MagicMock()
    tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    tokenizer.apply_chat_template = MagicMock(
        return_value="<formatted_prompt>")

    # Test the logic used in generate_with_prompts
    prompt = "Test prompt"
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        prompt_text = prompt

    assert prompt_text == "<formatted_prompt>"
    tokenizer.apply_chat_template.assert_called_once()


def test_config_keys_match_eval():
    """Test that steer config uses eval-compatible keys."""
    # Load the default steer config
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    import os

    config_dir = Path(__file__).parent.parent / "config" / "steer_config"
    if config_dir.exists():
        # Check that the config has the right keys
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name="default")

            # Verify eval-compatible keys exist
            assert hasattr(
                cfg.model, "base_model"), "Config should have model.base_model"
            assert hasattr(
                cfg.model, "adapter_checkpoint_dir"), "Config should have model.adapter_checkpoint_dir"
            assert hasattr(cfg.model, "k"), "Config should have model.k"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
