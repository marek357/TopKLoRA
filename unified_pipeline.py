#!/usr/bin/env python3
"""
Unified Interpretability Analysis Pipeline
Combines hypothesis generation and simulation validation for neural network latents.
Supports both local inference and API-based analysis.
"""

import json
import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
import logging
from datetime import datetime
import gc
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

# Optional imports for different inference backends
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True  
except ImportError:
    ANTHROPIC_AVAILABLE = False

from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_interpretability_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set memory allocation config for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


@dataclass
class SimulationExample:
    """Single example for simulation."""
    text: str
    tokenized_text: List[str]
    token_positions: List[int]
    actual_activations: Dict[int, float]  # position -> magnitude
    example_id: int


@dataclass
class SimulationPrediction:
    """Prediction for a single token."""
    token: str
    position: int
    predicted_activation: float  # 0-1 probability or magnitude
    confidence: float
    reasoning: Optional[str] = None


@dataclass
class SimulationResult:
    """Results of simulating a latent's behavior."""
    module_name: str
    latent_idx: int
    hypothesis: str
    correlation_score: float
    precision: float
    recall: float
    f1_score: float
    auc_score: float
    top_k_accuracy: float  # Accuracy in predicting top-k activations
    num_examples: int
    num_tokens: int
    detailed_metrics: Dict = field(default_factory=dict)
    examples_analyzed: List[Dict] = field(default_factory=list)


@dataclass
class InterpretabilityResult:
    module_name: str
    latent_idx: int
    hypothesis: str
    evidence: List[str]
    pattern_summary: Dict[str, str]
    confidence: str
    confidence_score: float  # For filtering
    alternative_interpretations: str
    uncertainties: str
    num_examples: int
    effect_rate: float
    timestamp: str
    model_used: str = "unknown"
    requires_api_review: bool = False
    simulation_result: Optional[SimulationResult] = None  # Added for validation


class UnifiedInterpretabilityAnalyzer:
    """Unified analyzer supporting multiple inference backends."""
    
    def __init__(
        self,
        backend: str = "vllm",  # "vllm", "openai", "anthropic"
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        gpu_devices: Optional[List[int]] = None,
        max_parallel_requests: int = 10,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        self.backend = backend
        self.gpu_devices = gpu_devices or [0]
        self.max_parallel_requests = max_parallel_requests
        
        if backend == "vllm":
            if not VLLM_AVAILABLE:
                raise ImportError("vLLM not installed. Install with: pip install vllm")
            self._init_vllm(model_name or "Qwen/Qwen2.5-32B-Instruct-AWQ", cache_dir, **kwargs)
        elif backend == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI client not installed. Install with: pip install openai")
            self._init_openai(model_name or "gpt-4-turbo-preview", api_key)
        elif backend == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic client not installed. Install with: pip install anthropic")
            self._init_anthropic(model_name or "claude-3-sonnet-20240229", api_key)
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
        logger.info(f"Initialized {backend} backend with model {self.model_name}")
        
    def _init_vllm(self, model_name: str, cache_dir: Optional[str], **kwargs):
        """Initialize vLLM backend."""
        self.model_name = model_name
        
        # Determine tensor parallel size based on available GPUs
        tensor_parallel_size = len(self.gpu_devices)
        
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.90),
            max_model_len=kwargs.get('max_model_len', 16384),
            download_dir=cache_dir,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
            quantization="awq" if "AWQ" in model_name else None,
            enforce_eager=True,
        )
        
        # Load tokenizer
        base_model = model_name.replace("-AWQ", "")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=2048,
            stop=["---", "\n\n\n"],
            repetition_penalty=1.05,
        )
        
    def _init_openai(self, model_name: str, api_key: Optional[str]):
        """Initialize OpenAI backend."""
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.tokenizer = None  # Will use tiktoken if needed
        
    def _init_anthropic(self, model_name: str, api_key: Optional[str]):
        """Initialize Anthropic backend."""
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.tokenizer = None
        
    async def _generate_api_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async generation for API backends."""
        if self.backend == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    # temperature=0.1,
                    max_completion_tokens=16384
                )
            )
            return response.choices[0].message.content
            
        elif self.backend == "anthropic":
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model_name,
                    max_tokens=2048,
                    temperature=0.1,
                    system=system_prompt or "You are an expert AI researcher analyzing neural network components.",
                    messages=[{"role": "user", "content": prompt}]
                )
            )
            return response.content[0].text
            
    def generate(self, prompts: Union[str, List[str]], system_prompt: Optional[str] = None) -> List[str]:
        """Generate responses for prompts using configured backend."""
        if isinstance(prompts, str):
            prompts = [prompts]
            
        if self.backend == "vllm":
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [output.outputs[0].text for output in outputs]
            
        else:  # API backends
            async def run_batch():
                semaphore = asyncio.Semaphore(self.max_parallel_requests)
                
                async def generate_with_limit(prompt):
                    async with semaphore:
                        return await self._generate_api_async(prompt, system_prompt)
                        
                tasks = [generate_with_limit(prompt) for prompt in prompts]
                return await asyncio.gather(*tasks)
                
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(run_batch())
            finally:
                loop.close()


def format_example_with_activation_tags(example, tokenizer):
    """Format the full input with inline activation magnitude tags."""
    tokens = example.tokenised_input_text
    activation_map = dict(zip(example.activating_idx, example.activation_magnitude))
    
    formatted_text = ""
    for idx, token in enumerate(tokens):
        if idx in activation_map:
            magnitude = activation_map[idx]
            token_str = tokenizer.convert_tokens_to_string([token]) if tokenizer else token
            formatted_text += f" [{token_str.strip()}|{magnitude:.2f}]"
        else:
            print('heree', token)
            token_str = tokenizer.convert_tokens_to_string([token]) if tokenizer else token
            formatted_text += token_str
    
    return formatted_text

def format_example_with_activation_tags_fastest(example, tokenizer):
    """Fastest version using batch operations and pre-computation."""
    tokens = example.tokenised_input_text
    activation_indices = example.activating_idx
    activation_magnitudes = example.activation_magnitude
    
    # Early exit if no activations
    if not activation_indices:
        return tokenizer.convert_tokens_to_string(tokens) if tokenizer else ''.join(tokens)
    
    # Create activation lookup (faster than dict for small sets)
    activation_map = dict(zip(activation_indices, activation_magnitudes))
    
    # Pre-allocate result list
    parts = []
    
    # Process in chunks between activations for efficiency
    last_idx = 0
    
    for idx in sorted(activation_indices):
        # Add non-activated tokens before this activation
        if idx > last_idx:
            chunk = tokens[last_idx:idx]
            if tokenizer:
                parts.append(tokenizer.convert_tokens_to_string(chunk))
            else:
                parts.extend(chunk)
        
        # Add activated token
        token = tokens[idx]
        magnitude = activation_map[idx]
        if tokenizer:
            token_str = tokenizer.convert_tokens_to_string([token]).strip()
        else:
            token_str = str(token)
        parts.append(f" [{token_str}|{magnitude:.2f}]")
        
        last_idx = idx + 1
    
    # Add remaining tokens
    if last_idx < len(tokens):
        chunk = tokens[last_idx:]
        if tokenizer:
            parts.append(tokenizer.convert_tokens_to_string(chunk))
        else:
            parts.extend(chunk)
    
    return ''.join(parts)


def generate_autointerp_prompt(latent_data, module_name, latent_idx, tokenizer, max_examples=10):
    """Generate a comprehensive interpretability prompt for a specific latent."""
    
    # Filter to only examples with causal effects
    examples_with_effects = [ex for ex in latent_data['examples'] if ex.ablation_has_effect]
    
    # Handle case where latent has no causal effects
    if not examples_with_effects:
        return f"""## Non-Causal Latent Detected

You are analyzing latent {latent_idx} in module {module_name} from a TopK LoRA adapter.

**Key Finding**: This latent activated {latent_data['num_activations']} times across the dataset, but ablating it NEVER changed the model's output.

This suggests the latent is redundant, only matters in combination with other latents, encodes non-behavioral information, or is undertrained.

No further interpretation is needed for this non-causal latent."""
    
    # Build prompt for causal latents
    prompt = f"""You are analyzing a latent component in a TopK LoRA adapter from a language model.

## Latent Information
- Module: {module_name}
- Latent Index: {latent_idx}
- Total Activations Found: {latent_data['num_activations']}
- Causal Examples: {len(examples_with_effects)} (out of {len(latent_data['examples'])} tested)
- Causal Effect Rate: {latent_data['effect_rate']:.1%}

## Causal Examples

Below are examples where this latent both activated strongly AND causally influenced the model's output when disabled:

"""
    
    # Sort examples by activation magnitude
    examples = sorted(examples_with_effects[:max_examples], 
                     key=lambda x: max(abs(m) for m in x.activation_magnitude), 
                     reverse=True)
    
    for i, example in tqdm(enumerate(examples, 1)):
        prompt += f"\n### Example {i}\n"
        
        # Show statistics
        max_magnitude = max(example.activation_magnitude, key=abs)
        num_activations = len(example.activation_magnitude)
        
        prompt += f"**Activation Statistics:**\n"
        prompt += f"- Number of activating tokens: {num_activations}\n"
        prompt += f"- Strongest activation: {max_magnitude:.2f}\n"
        prompt += f"- Activating tokens: {example.activating_token}\n"
        
        # Show full input with activation tags
        prompt += f"\n**Full Input (with activation magnitudes):**\n```\n"

        # prompt += format_example_with_activation_tags(example, tokenizer)
        prompt += format_example_with_activation_tags_fastest(example, tokenizer)
        prompt += f"\n```\n"
        
        # Show output change
        prompt += f"\n**Output Change When Latent Disabled:**\n"
        baseline_words = example.baseline_text.split()
        ablated_words = example.ablated_text.split()
        
        diverge_idx = next((j for j, (b, a) in enumerate(zip(baseline_words, ablated_words)) if b != a), len(baseline_words))
        
        if diverge_idx < len(baseline_words):
            context_start = max(0, diverge_idx - 25)
            context_end = min(max(len(baseline_words), len(ablated_words)), diverge_idx + 30)
            
            baseline_context = " ".join(baseline_words[context_start:context_end])
            ablated_context = " ".join(ablated_words[context_start:context_end] if context_start < len(ablated_words) else ["[GENERATION ENDED]"])
            
            prompt += f"- **Original**: ...{baseline_context}...\n"
            prompt += f"- **Ablated**: ...{ablated_context}...\n"
        
        prompt += "\n---\n"
    
    # Add interpretation instructions
    prompt += """
## Interpretation Task

Based on the causal examples above, form a hypothesis about what this latent represents and how it functions. Consider:
1. Input patterns (semantic, syntactic, or pragmatic)
2. Functional role (how disabling changes behavior)
3. Contextual patterns
4. Magnitude patterns

## Required Output Format

**Hypothesis**: [1-2 sentence description of what this latent encodes and its functional role]

**Evidence**: 
- [Key observations with example references]

**Pattern Summary**:
- Input patterns: [What activates this latent]
- Output effects: [How behavior changes when disabled]
- Activation characteristics: [Magnitude patterns]

**Confidence**: [low/medium/high] - with justification

**Alternative Interpretations**: [Other plausible interpretations]

**Uncertainties**: [What needs more investigation]
"""
    
    return prompt


def create_simulation_prompt(hypothesis: str, example: SimulationExample, include_cot: bool = True) -> str:
    """Create prompt for simulating latent activation."""
    
    prompt = f"""You are simulating a neural network latent that has been interpreted as:

**Latent Hypothesis**: {hypothesis}

Your task is to predict which tokens in the following text would activate this latent.

**Text to analyze**:
{example.text}

**Tokenized version** (each token with its position):
"""
    
    for pos, token in enumerate(example.tokenized_text):
        prompt += f"{pos}: {token}\n"
    
    if include_cot:
        prompt += """

Analyze each token and predict whether it would activate the latent based on the hypothesis.

For each activating token, provide:
- Position
- Activation strength (0.0-1.0)
- Brief reasoning

Format your response as:
ANALYSIS:
[Your analysis]

PREDICTIONS:
Position | Token | Activation | Confidence | Reasoning
[List predictions]

NO_ACTIVATION:
[List non-activating positions: pos1, pos2, ...]
"""
    else:
        prompt += """

List tokens that would activate this latent.

PREDICTIONS:
Position | Token | Activation | Confidence
[List predictions]

NO_ACTIVATION:
[List non-activating positions]
"""
    
    return prompt


def parse_simulation_response(response: str, num_tokens: int) -> Dict[int, float]:
    """Parse model's predictions from response."""
    predictions = {}
    
    try:
        # Extract predictions section
        if "PREDICTIONS:" in response:
            pred_start = response.find("PREDICTIONS:") + len("PREDICTIONS:")
            pred_end = response.find("NO_ACTIVATION:", pred_start)
            if pred_end == -1:
                pred_end = len(response)
            
            predictions_text = response[pred_start:pred_end].strip()
            
            # Parse each prediction line
            for line in predictions_text.split('\n'):
                if '|' in line and not line.startswith('Position'):
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 3:
                        try:
                            position = int(parts[0])
                            activation = float(parts[2])
                            predictions[position] = min(max(activation, 0.0), 1.0)
                        except (ValueError, IndexError):
                            continue
        
        # Extract no activation positions
        if "NO_ACTIVATION:" in response:
            no_act_start = response.find("NO_ACTIVATION:") + len("NO_ACTIVATION:")
            no_act_text = response[no_act_start:].strip()
            
            # Handle both list and range formats
            if no_act_text.startswith('['):
                no_act_text = no_act_text.strip('[]')
            
            # Parse positions
            for pos_str in no_act_text.split(','):
                pos_str = pos_str.strip()
                if '-' in pos_str:  # Range
                    try:
                        start, end = map(int, pos_str.split('-'))
                        for pos in range(start, end + 1):
                            if 0 <= pos < num_tokens:
                                predictions[pos] = 0.0
                    except ValueError:
                        continue
                else:  # Single position
                    try:
                        pos = int(pos_str)
                        if 0 <= pos < num_tokens:
                            predictions[pos] = 0.0
                    except ValueError:
                        continue
        
        # Fill in missing positions with 0
        for i in range(num_tokens):
            if i not in predictions:
                predictions[i] = 0.0
                
    except Exception as e:
        logger.warning(f"Error parsing simulation response: {e}")
        predictions = {i: 0.0 for i in range(num_tokens)}
    
    return predictions


def compute_simulation_metrics(predictions: List[Dict[int, float]], 
                             examples: List[SimulationExample],
                             activation_threshold: float = 1.0) -> Dict:
    """Compute correlation and accuracy metrics."""
    all_predicted = []
    all_actual = []
    all_binary_predicted = []
    all_binary_actual = []
    
    for preds, example in zip(predictions, examples):
        for pos in range(len(example.tokenized_text)):
            pred_val = preds.get(pos, 0.0)
            actual_val = abs(example.actual_activations.get(pos, 0.0))
            
            all_predicted.append(pred_val)
            all_actual.append(actual_val)
            
            all_binary_predicted.append(pred_val > 0.5)
            all_binary_actual.append(actual_val > activation_threshold)
    
    # Convert to numpy arrays
    all_predicted = np.array(all_predicted)
    all_actual = np.array(all_actual)
    all_binary_predicted = np.array(all_binary_predicted)
    all_binary_actual = np.array(all_binary_actual)
    
    # Compute metrics
    pearson_corr, _ = pearsonr(all_predicted, all_actual)
    spearman_corr, _ = spearmanr(all_predicted, all_actual)
    
    # Classification metrics
    precision, recall, _ = precision_recall_curve(all_binary_actual, all_predicted)
    pr_auc = auc(recall, precision)
    
    try:
        roc_auc = roc_auc_score(all_binary_actual, all_predicted)
    except ValueError:
        roc_auc = 0.5
    
    # F1 score
    true_positives = np.sum(all_binary_predicted & all_binary_actual)
    false_positives = np.sum(all_binary_predicted & ~all_binary_actual)
    false_negatives = np.sum(~all_binary_predicted & all_binary_actual)
    
    precision_at_threshold = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall_at_threshold = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision_at_threshold * recall_at_threshold) / (precision_at_threshold + recall_at_threshold) if (precision_at_threshold + recall_at_threshold) > 0 else 0
    
    return {
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'precision': precision_at_threshold,
        'recall': recall_at_threshold,
        'f1_score': f1,
        'total_actual_activations': int(np.sum(all_binary_actual)),
        'total_predicted_activations': int(np.sum(all_binary_predicted))
    }


class UnifiedPipeline:
    """Main pipeline combining interpretation and simulation."""
    
    def __init__(self, analyzer: UnifiedInterpretabilityAnalyzer, tokenizer):
        self.analyzer = analyzer
        self.tokenizer = tokenizer
        
    def parse_interpretation_response(self, response: str) -> Dict:
        """Parse the structured interpretation response."""
        result = {
            'hypothesis': '',
            'evidence': [],
            'pattern_summary': {
                'input_patterns': '',
                'output_effects': '',
                'activation_characteristics': ''
            },
            'confidence': 'low',
            'confidence_score': 0.0,
            'alternative_interpretations': '',
            'uncertainties': ''
        }
        
        try:
            # Extract sections using string matching
            sections = {
                'Hypothesis': 'hypothesis',
                'Evidence': 'evidence',
                'Pattern Summary': 'pattern_summary',
                'Confidence': 'confidence',
                'Alternative Interpretations': 'alternative_interpretations',
                'Uncertainties': 'uncertainties'
            }
            
            for section_name, field_name in sections.items():
                if f"**{section_name}**:" in response:
                    start = response.find(f"**{section_name}**:") + len(f"**{section_name}**:")
                    # Find next section
                    next_sections = [response.find(f"**{s}**:", start) for s in sections if response.find(f"**{s}**:", start) > start]
                    end = min(next_sections) if next_sections else len(response)
                    
                    content = response[start:end].strip()
                    
                    if field_name == 'evidence':
                        result['evidence'] = [e.strip()[1:].strip() for e in content.split('\n') if e.strip().startswith('-')]
                    elif field_name == 'pattern_summary':
                        for pattern in ['Input patterns:', 'Output effects:', 'Activation characteristics:']:
                            if pattern in content:
                                pattern_start = content.find(pattern) + len(pattern)
                                pattern_end = content.find('\n-', pattern_start)
                                if pattern_end == -1:
                                    pattern_end = len(content)
                                key = pattern.replace(':', '').lower().replace(' ', '_')
                                result['pattern_summary'][key] = content[pattern_start:pattern_end].strip()
                    elif field_name == 'confidence':
                        conf_text = content.lower()
                        if "high" in conf_text:
                            result['confidence'] = 'high'
                            result['confidence_score'] = 0.8
                        elif "medium" in conf_text:
                            result['confidence'] = 'medium'
                            result['confidence_score'] = 0.5
                        else:
                            result['confidence'] = 'low'
                            result['confidence_score'] = 0.2
                    else:
                        result[field_name] = content
                        
        except Exception as e:
            logger.warning(f"Error parsing interpretation response: {e}")
            
        return result
        
    def analyze_single_latent(
        self,
        module_name: str,
        latent_idx: int,
        ablation_results: Dict,
        run_simulation: bool = True,
        num_simulation_examples: int = 50
    ) -> InterpretabilityResult:
        """Analyze a single latent with optional simulation validation."""
        
        # Get latent data
        latent_data = ablation_results[module_name][latent_idx]

        logger.info('Generating prompt')
        
        # Generate interpretation prompt
        interp_prompt = generate_autointerp_prompt(
            latent_data=latent_data,
            module_name=module_name,
            latent_idx=latent_idx,
            tokenizer=self.tokenizer,
            max_examples=20
        )
        
        logger.info(f'Prompt generated {interp_prompt}')
        # Get interpretation
        logger.info(f"Interpreting {module_name}/{latent_idx}")
        response = self.analyzer.generate(interp_prompt)[0]
        print('this is the response:')
        print(response)
        parsed = self.parse_interpretation_response(response)
        
        # Create result
        result = InterpretabilityResult(
            module_name=module_name,
            latent_idx=latent_idx,
            hypothesis=parsed['hypothesis'],
            evidence=parsed['evidence'],
            pattern_summary=parsed['pattern_summary'],
            confidence=parsed['confidence'],
            confidence_score=parsed['confidence_score'],
            alternative_interpretations=parsed['alternative_interpretations'],
            uncertainties=parsed['uncertainties'],
            num_examples=len(latent_data.get('examples', [])),
            effect_rate=latent_data.get('effect_rate', 0),
            timestamp=datetime.now().isoformat(),
            model_used=self.analyzer.model_name,
            requires_api_review=(parsed['confidence_score'] < 0.5)
        )
        
        # Run simulation if requested
        if run_simulation and parsed['hypothesis']:
            logger.info(f"Running simulation validation for {module_name}/{latent_idx}")
            simulation_result = self.run_simulation_validation(
                hypothesis=parsed['hypothesis'],
                module_name=module_name,
                latent_idx=latent_idx,
                ablation_results=ablation_results,
                num_examples=num_simulation_examples
            )
            result.simulation_result = simulation_result
            
        return result
        
    def run_simulation_validation(
        self,
        hypothesis: str,
        module_name: str,
        latent_idx: int,
        ablation_results: Dict,
        num_examples: int = 50,
        batch_size: int = 4
    ) -> SimulationResult:
        """Run simulation validation for a hypothesis."""
        
        # Prepare examples
        examples = self.prepare_simulation_examples(
            ablation_results=ablation_results,
            module_name=module_name,
            latent_idx=latent_idx,
            num_examples=num_examples
        )
        
        if len(examples) < 10:
            logger.warning(f"Insufficient examples for simulation: {len(examples)}")
            return None
            
        # Generate simulation prompts
        all_predictions = []
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            prompts = [
                create_simulation_prompt(hypothesis, ex, include_cot=True)
                for ex in batch
            ]
            
            # Get predictions
            responses = self.analyzer.generate(prompts)
            
            # Parse predictions
            for response, example in zip(responses, batch):
                predictions = parse_simulation_response(
                    response,
                    len(example.tokenized_text)
                )
                all_predictions.append(predictions)
        
        # Compute metrics
        metrics = compute_simulation_metrics(all_predictions, examples)
        
        # Create result
        return SimulationResult(
            module_name=module_name,
            latent_idx=latent_idx,
            hypothesis=hypothesis,
            correlation_score=metrics['pearson_correlation'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            auc_score=metrics['pr_auc'],
            top_k_accuracy=0.0,  # Not computed in simplified version
            num_examples=len(examples),
            num_tokens=sum(len(ex.tokenized_text) for ex in examples),
            detailed_metrics=metrics
        )
        
    def prepare_simulation_examples(
        self,
        ablation_results: Dict,
        module_name: str,
        latent_idx: int,
        num_examples: int = 50
    ) -> List[SimulationExample]:
        """Prepare examples for simulation from ablation results."""
        examples = []
        
        latent_data = ablation_results[module_name][latent_idx]
        available_examples = latent_data['examples']
        
        # Use test set (examples not used for interpretation)
        test_examples = available_examples[10:] if len(available_examples) > 20 else available_examples
        sampled = test_examples[:num_examples]
        
        for ex in sampled:
            activation_map = {}
            for pos, magnitude in zip(ex.activating_idx, ex.activation_magnitude):
                activation_map[pos] = magnitude
            
            sim_example = SimulationExample(
                text=ex.input_text,
                tokenized_text=ex.tokenised_input_text,
                token_positions=list(range(len(ex.tokenised_input_text))),
                actual_activations=activation_map,
                example_id=ex.example_idx
            )
            
            examples.append(sim_example)
        
        return examples
        
    def analyze_module_parallel(
        self,
        module_name: str,
        ablation_results: Dict,
        min_effect_rate: float = 0.1,
        max_latents: Optional[int] = None,
        run_simulation: bool = True,
        batch_size: int = 8
    ) -> List[InterpretabilityResult]:
        """Analyze all latents in a module with parallel processing."""
        
        module_results = ablation_results.get(module_name, {})
        
        # Filter and sort latents
        causal_latents = [
            (idx, data) for idx, data in module_results.items()
            if data.get('effect_rate', 0) >= min_effect_rate
        ]
        causal_latents.sort(key=lambda x: x[1]['effect_rate'], reverse=True)
        
        if max_latents:
            causal_latents = causal_latents[:max_latents]
            
        logger.info(f"Analyzing {len(causal_latents)} latents in {module_name}")
        
        # Prepare all prompts
        prompts_and_metadata = []
        for latent_idx, latent_data in causal_latents:
            prompt = generate_autointerp_prompt(
                latent_data=latent_data,
                module_name=module_name,
                latent_idx=latent_idx,
                tokenizer=self.tokenizer,
                max_examples=20
            )
            
            metadata = {
                'module_name': module_name,
                'latent_idx': latent_idx,
                'latent_data': latent_data,
                'num_examples': len(latent_data.get('examples', [])),
                'effect_rate': latent_data.get('effect_rate', 0)
            }
            
            prompts_and_metadata.append((prompt, metadata))
        
        # Process in batches
        all_results = []
        
        for i in tqdm(range(0, len(prompts_and_metadata), batch_size), desc=f"Analyzing {module_name}"):
            batch = prompts_and_metadata[i:i + batch_size]
            batch_prompts = [p[0] for p in batch]
            batch_metadata = [p[1] for p in batch]
            
            # Get interpretations
            try:
                responses = self.analyzer.generate(batch_prompts)
            except Exception as e:
                logger.error(f"Error in batch {i}: {e}")
                continue
                
            # Process responses
            for response, metadata in zip(responses, batch_metadata):
                parsed = self.parse_interpretation_response(response)
                
                result = InterpretabilityResult(
                    module_name=metadata['module_name'],
                    latent_idx=metadata['latent_idx'],
                    hypothesis=parsed['hypothesis'],
                    evidence=parsed['evidence'],
                    pattern_summary=parsed['pattern_summary'],
                    confidence=parsed['confidence'],
                    confidence_score=parsed['confidence_score'],
                    alternative_interpretations=parsed['alternative_interpretations'],
                    uncertainties=parsed['uncertainties'],
                    num_examples=metadata['num_examples'],
                    effect_rate=metadata['effect_rate'],
                    timestamp=datetime.now().isoformat(),
                    model_used=self.analyzer.model_name,
                    requires_api_review=(parsed['confidence_score'] < 0.5)
                )
                
                # Run simulation if requested
                if run_simulation and parsed['hypothesis']:
                    simulation_result = self.run_simulation_validation(
                        hypothesis=parsed['hypothesis'],
                        module_name=metadata['module_name'],
                        latent_idx=metadata['latent_idx'],
                        ablation_results=ablation_results,
                        num_examples=50
                    )
                    result.simulation_result = simulation_result
                    
                all_results.append(result)
                
            # Clear memory
            if self.analyzer.backend == "vllm":
                torch.cuda.empty_cache()
                gc.collect()
                
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Unified Interpretability Analysis Pipeline")
    
    # Input/output
    parser.add_argument("--ablation-results", required=True, help="Path to ablation results pickle")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    # Analysis options
    parser.add_argument("--single-latent", nargs=2, metavar=("MODULE", "INDEX"),
                       help="Analyze single latent: module_name latent_index")
    parser.add_argument("--min-effect-rate", type=float, default=0.1,
                       help="Minimum effect rate to analyze")
    parser.add_argument("--max-latents-per-module", type=int,
                       help="Maximum latents to analyze per module")
    parser.add_argument("--run-simulation", action="store_true",
                       help="Run simulation validation after interpretation")
    
    # Backend configuration
    parser.add_argument("--backend", choices=["vllm", "openai", "anthropic"], default="vllm",
                       help="Inference backend to use")
    parser.add_argument("--model", help="Model name for the backend")
    parser.add_argument("--api-key", help="API key for cloud backends")
    parser.add_argument("--tokenizer-path", required=True,
                       help="Path to tokenizer (for prompt formatting)")
    
    # Processing options
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs to use (for vLLM)")
    parser.add_argument("--gpu-devices", nargs="+", type=int,
                       help="Specific GPU device IDs to use")
    parser.add_argument("--max-parallel-requests", type=int, default=10,
                       help="Max parallel API requests")
    
    args = parser.parse_args()
    
    # Setup
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Initialize analyzer
    gpu_devices = args.gpu_devices or list(range(args.num_gpus))
    
    analyzer = UnifiedInterpretabilityAnalyzer(
        backend=args.backend,
        model_name=args.model,
        api_key=args.api_key,
        gpu_devices=gpu_devices,
        max_parallel_requests=args.max_parallel_requests
    )
    
    # Create pipeline
    pipeline = UnifiedPipeline(analyzer, tokenizer)
    
    # Load ablation results
    logger.info(f"Loading ablation results from {args.ablation_results}")
    with open(args.ablation_results, 'rb') as f:
        ablation_results = pickle.load(f)
    
    # Run analysis
    if args.single_latent:
        # Single latent analysis
        module_name, latent_idx = args.single_latent[0], int(args.single_latent[1])
        logger.info(f"Analyzing single latent: {module_name}/{latent_idx}")
        
        result = pipeline.analyze_single_latent(
            module_name=module_name,
            latent_idx=latent_idx,
            ablation_results=ablation_results,
            run_simulation=args.run_simulation
        )
        
        # Save result
        output_file = output_path / f"latent_{module_name.replace('/', '_')}_{latent_idx}.json"
        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
            
        logger.info(f"Saved result to {output_file}")
        
        # Print summary
        print(f"\nAnalysis Results for {module_name}/{latent_idx}:")
        print(f"Hypothesis: {result.hypothesis}")
        print(f"Confidence: {result.confidence}")
        if result.simulation_result:
            print(f"Simulation F1 Score: {result.simulation_result.f1_score:.3f}")
            
    else:
        # Full analysis
        all_results = []
        
        for module_name in tqdm(ablation_results.keys(), desc="Processing modules"):
            # Check for checkpoint
            checkpoint_file = output_path / f"checkpoint_{module_name.replace('/', '_')}.json"
            if checkpoint_file.exists():
                logger.info(f"Loading checkpoint for {module_name}")
                with open(checkpoint_file, 'r') as f:
                    module_results = [InterpretabilityResult(**r) for r in json.load(f)]
                all_results.extend(module_results)
                continue
                
            # Analyze module
            module_results = pipeline.analyze_module_parallel(
                module_name=module_name,
                ablation_results=ablation_results,
                min_effect_rate=args.min_effect_rate,
                max_latents=args.max_latents_per_module,
                run_simulation=args.run_simulation,
                batch_size=args.batch_size
            )
            
            # Save checkpoint
            with open(checkpoint_file, 'w') as f:
                json.dump([asdict(r) for r in module_results], f, indent=2)
                
            all_results.extend(module_results)
            
        # Save final results
        final_output = output_path / "interpretability_results.json"
        with open(final_output, 'w') as f:
            json.dump([asdict(r) for r in all_results], f, indent=2)
            
        # Generate summary report
        generate_summary_report(all_results, output_path)
        
        logger.info(f"\nAnalysis complete!")
        logger.info(f"Total latents analyzed: {len(all_results)}")
        logger.info(f"Results saved to {final_output}")


def generate_summary_report(results: List[InterpretabilityResult], output_path: Path):
    """Generate a summary report of the analysis."""
    report = {
        'total_analyzed': len(results),
        'by_confidence': {
            'high': sum(1 for r in results if r.confidence == 'high'),
            'medium': sum(1 for r in results if r.confidence == 'medium'),
            'low': sum(1 for r in results if r.confidence == 'low')
        },
        'requiring_api_review': sum(1 for r in results if r.requires_api_review),
        'average_effect_rate': sum(r.effect_rate for r in results) / len(results) if results else 0,
        'modules_analyzed': len(set(r.module_name for r in results)),
        'timestamp': datetime.now().isoformat()
    }
    
    # Add simulation results if available
    sim_results = [r for r in results if r.simulation_result]
    if sim_results:
        report['simulation_validation'] = {
            'num_validated': len(sim_results),
            'avg_f1_score': sum(r.simulation_result.f1_score for r in sim_results) / len(sim_results),
            'avg_correlation': sum(r.simulation_result.correlation_score for r in sim_results) / len(sim_results),
            'high_quality': sum(1 for r in sim_results if r.simulation_result.f1_score > 0.7)
        }
    
    # Group by module
    by_module = defaultdict(list)
    for r in results:
        by_module[r.module_name].append(r)
    
    report['by_module'] = {
        module: {
            'total': len(module_results),
            'high_confidence': sum(1 for r in module_results if r.confidence == 'high'),
            'avg_effect_rate': sum(r.effect_rate for r in module_results) / len(module_results)
        }
        for module, module_results in by_module.items()
    }
    
    # Save report
    with open(output_path / "analysis_summary.json", 'w') as f:
        json.dump(report, f, indent=2)
        
    # Save high-quality results separately
    high_quality = [r for r in results if r.confidence == 'high']
    if high_quality:
        with open(output_path / "high_confidence_results.json", 'w') as f:
            json.dump([asdict(r) for r in high_quality], f, indent=2)
    
    logger.info(f"Generated summary report at {output_path / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()