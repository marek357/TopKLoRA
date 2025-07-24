#!/usr/bin/env python3
"""
LoRA Latent Interpreter - Generates and validates hypotheses about sparse LoRA adapter latents
Based on methodology from OpenAI's neuron explainer paper
"""

import argparse
import json
import pickle
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
import numpy as np
from scipy import stats
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import dotenv
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

@dataclass
class LatentActivation:
    """Matches the structure from your code"""
    example_idx: int
    activation_magnitude: List[float]
    activating_token_id: List[int]
    activating_idx: List[int]
    activating_idx_padded: List[int]
    activating_token: List[Optional[str]]
    is_padding_token: List[Optional[bool]]
    baseline_text: Optional[str] = None
    ablated_text: Optional[str] = None
    input_text: Optional[str] = None
    collated_input_text: Optional[str] = None
    tokenised_input_text: List[Optional[str]] = None
    ablation_has_effect: Optional[bool] = None


@dataclass
class ActivationContext:
    """Context around an activation for LLM analysis"""
    text_before: str
    activating_tokens: List[str]
    text_after: str
    activation_strength: float
    token_position: int
    example_idx: int


@dataclass
class Hypothesis:
    """A hypothesis about what a latent represents"""
    description: str
    confidence: float
    examples_used: List[int]
    timestamp: str


@dataclass
class ValidationResult:
    """Results from validating a hypothesis"""
    hypothesis: Hypothesis
    correlation: float
    precision_at_threshold: float
    recall_at_threshold: float
    false_positive_examples: List[str]
    false_negative_examples: List[str]


class LLMInterface(ABC):
    """Abstract interface for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        pass
    
    @abstractmethod
    def generate_json(self, prompt: str, max_tokens: int = 1000) -> Dict:
        pass


class OpenAIInterface(LLMInterface):
    """OpenAI API interface"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1.0
        )
        return response.choices[0].message.content
    
    def generate_json(self, prompt: str, max_tokens: int = 8000) -> Dict:
        # print(prompt)
        # assert False
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=1.0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


class VLLMInterface(LLMInterface):
    """vLLM interface for local models"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-32B-Instruct-AWQ", base_url: str = "http://localhost:8000"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="dummy"  # vLLM doesn't need a real key
        )
        self.model = model_name
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    
    def generate_json(self, prompt: str, max_tokens: int = 1000) -> Dict:
        # For models that don't support JSON mode, we'll parse the output
        json_prompt = prompt + "\n\nPlease respond with valid JSON only."
        response = self.generate(json_prompt, max_tokens)
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON, returning empty dict")
                return {}
        return {}


class LatentInterpreter:
    """Main class for interpreting LoRA latents"""
    
    def __init__(self, 
                 llm: LLMInterface,
                 tokenizer_name: str = "google/gemma-2-2b",
                 context_window: int = 20,
                 top_k_examples: int = 20,
                 cache_dir: str = "cache/interpretations"):
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.context_window = context_window
        self.top_k_examples = top_k_examples
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)


    def calibrate_predictions(self, predicted_scores: List[float], 
                            actual_activations: List[float]) -> Tuple[List[float], float]:
        """
        Calibrate predicted [0, 1] scores to match actual activation distribution.
        Based on OpenAI's neuron explainer methodology.
        """
        if len(predicted_scores) != len(actual_activations):
            raise ValueError(f"Predictions ({len(predicted_scores)}) and actuals ({len(actual_activations)}) must have same length")
        
        if len(predicted_scores) == 0:
            return [], 0.0
        
        # Calculate correlation first
        correlation, _ = stats.pearsonr(predicted_scores, actual_activations)
        
        # Calculate statistics
        pred_mean = np.mean(predicted_scores)
        pred_std = np.std(predicted_scores)
        actual_mean = np.mean(actual_activations)
        actual_std = np.std(actual_activations)
        
        # Calibrate: shift mean and scale by correlation
        calibrated_predictions = []
        for pred in predicted_scores:
            # Center
            centered = pred - pred_mean
            # Scale by ρ * (actual_std / pred_std)
            if pred_std > 0:
                scaled = centered * (correlation * actual_std / pred_std)
            else:
                scaled = 0
            # Shift to actual mean
            calibrated = scaled + actual_mean
            calibrated_predictions.append(calibrated)
        
        return calibrated_predictions, correlation

    def create_token_simulation_prompt(self, hypothesis: str, 
                                    activation: LatentActivation,
                                    context_size: int = 50) -> str:
        """Create prompt for token-level activation prediction."""
        
        tokens = activation.tokenised_input_text
        
        prompt = f"""You are simulating a neural network component (latent) that has been interpreted as:

    **Latent Hypothesis**: {hypothesis}

    Your task is to predict which tokens in the following text would activate this latent (cause it to fire strongly).

    **Tokenized text** (showing {len(tokens)} tokens):
    """
        # Show tokens with positions
        for pos, token in enumerate(tokens):
            # Convert token to readable string
            token_str = self.tokenizer.convert_tokens_to_string([token]).strip()
            prompt += f"{pos}: {token_str}\n"
        
        prompt += """

    For each token, predict the activation strength based on whether it matches the pattern described in the hypothesis.

    Provide your predictions in JSON format:
    {
        "predictions": [
            {"position": 0, "token": "token_text", "activation": 0.0-1.0, "reason": "brief explanation"},
            {"position": 1, "token": "token_text", "activation": 0.0-1.0, "reason": "brief explanation"},
            ...
        ]
    }

    Note: Only include tokens that would activate (activation > 0). Tokens not listed are assumed to have 0 activation.
    You may include up to 50 activating tokens. Focus on the strongest activations.
    """
        
        return prompt

    def parse_token_predictions(self, response: Dict, num_tokens: int) -> Dict[int, float]:
        """Parse token-level predictions from JSON response."""
        predictions = {i: 0.0 for i in range(num_tokens)}  # Initialize all to 0
        
        try:
            if "predictions" in response:
                for pred in response["predictions"]:
                    pos = pred.get("position", -1)
                    activation = pred.get("activation", 0.0)
                    
                    if 0 <= pos < num_tokens:
                        # Ensure activation is in [0, 1] range
                        predictions[pos] = max(0.0, min(1.0, abs(float(activation))))
                        
        except Exception as e:
            logger.warning(f"Error parsing token predictions: {e}")
        
        return predictions


    def generate_autointerp_prompt(self, activation_examples, module_name, latent_idx, max_examples=10):
        """
        Generate a comprehensive interpretability prompt for a specific TopK LoRA latent.
        Only includes examples where ablation has a causal effect on the output.
        
        Args:
            latent_data: Dictionary containing 'num_activations', 'examples', and 'effect_rate'
            module_name: Name of the module containing this latent
            latent_idx: Index of the latent being analyzed
            tokenizer: Tokenizer for decoding tokens
            max_examples: Maximum number of examples to include
        """
                 
        # Build prompt for causal latents
        prompt = f"""You are analyzing a latent component in a TopK LoRA adapter from a language model. This latent is part of a low-rank decomposition where only the top-k most active components are used during each forward pass.

    ## Latent Information
    - Module: {module_name}
    - Latent Index: {latent_idx}

    ## Examples
    Your task is to identify the common pattern or concept that this latent represents.
    For each example below, the full text is shown with activation magnitudes inline as [token|magnitude].

    """
        
        # Sort examples by activation magnitude
        examples = sorted(activation_examples[:max_examples], 
                        key=lambda x: max(abs(m) for m in x.activation_magnitude), 
                        reverse=True)
        
        for i, example in enumerate(examples, 1):
            prompt += f"\n### Example {i}\n"
            
            # Show statistics for this example
            max_magnitude = max(example.activation_magnitude, key=abs)
            num_activations = len(example.activation_magnitude)
            
            prompt += f"**Activation Statistics:**\n"
            prompt += f"- Number of activating tokens: {num_activations}\n"
            prompt += f"- Strongest activation: {max_magnitude:.2f}\n"
            prompt += f"- Activation range: [{min(example.activation_magnitude):.2f}, {max(example.activation_magnitude):.2f}]\n"
            prompt += f"- Activating tokens: {example.activating_token}\n"
            
            # Show full input with activation tags
            prompt += f"\n**Full Input (with activation magnitudes):**\n```\n"
            prompt += self.format_example_with_activation_tags(example)
            prompt += f"\n```\n"
            prompt += "\n---\n"
        
        # Add detailed interpretation instructions
        prompt += """
    ## Interpretation Task

    Based on the examples above, form a hypothesis about what this latent represents and how it functions in the model. Consider the following aspects:

    ### 1. **Input Pattern Analysis**
    - What do the tokens with high activation magnitudes have in common?
    - Are there semantic patterns (meaning-based), syntactic patterns (grammar-based), or pragmatic patterns (context/discourse-based)?
    - Do activations cluster around specific parts of the input (e.g., certain types of phrases, questions, statements)?
    - Are activation magnitudes consistent across similar tokens, or do they vary based on context?

    ### 3. **Contextual Patterns**
    - In what types of conversations or text does this latent activate?
    - Does it respond to specific topics, interaction patterns, or discourse structures?
    - Are there common scenarios across the examples?

    ### 4. **Magnitude Patterns**
    - Do certain contexts trigger stronger activations?
    - Is there a relationship between activation strength and the size of the effect when ablated?
    - Are negative and positive activations associated with different functions?

    ## Important Notes
    - Focus on patterns that appear across multiple examples
    - Be specific about which examples support each part of your interpretation
    - If activation magnitude patterns suggest different "modes" of the latent, describe them
    
    ## Required Output Format

    Please provide your analysis in the following JSON format:

    {
        "hypothesis": "A clear, concise 1-2 sentence description of what this latent encodes or detects, and its functional role",
        "pattern_type": "lexical|semantic|syntactic|contextual",
        "confidence": 0.0-1.0,
        "supporting_evidence": ["observation1", "observation2", ...]
    }

    """
        
        return prompt



    def validate_hypothesis_token_level(self, hypothesis: Hypothesis,
                                    test_activations: List[LatentActivation],
                                    latent_id: int,
                                    layer_name: str,
                                    max_examples: int = 5) -> ValidationResult:
        """
        Validate hypothesis using token-level correlation like OpenAI.
        """
        all_predicted_scores = []
        all_actual_scores = []
        all_actual_scores_shifted = []
        
        # Process each example
        for i, activation in enumerate(test_activations[:max_examples]):
            # Create token-level prompt
            prompt = self.create_token_simulation_prompt(
                # hypothesis.description,
                hypothesis,
                activation
            )
            print(prompt)
            
            # Get predictions
            try:
                response = self.llm.generate_json(prompt, max_tokens=8000)
                # print(response)
                token_predictions = self.parse_token_predictions(
                    response, 
                    len(activation.tokenised_input_text)
                )
                # print(token_predictions)
                
                # Create actual activation map
                actual_map = {pos: 0.0 for pos in range(len(activation.tokenised_input_text))}

                actual_map_shifted = {pos: 0.0 for pos in range(len(activation.tokenised_input_text))}
                for pos, mag in zip(activation.activating_idx, activation.activation_magnitude):
                    actual_map[pos-1] = abs(mag)
                    actual_map_shifted[pos] = abs(mag)
                print(f"Actual activation map for example {i}: {actual_map}")
                assert False
                # print(actual_map)
                # assert False
                # Collect predictions and actuals for all tokens
                for pos in range(len(activation.tokenised_input_text)):
                    all_predicted_scores.append(token_predictions.get(pos, 0.0))
                    all_actual_scores.append(actual_map.get(pos, 0.0))
                    all_actual_scores_shifted.append(actual_map_shifted.get(pos, 0.0))
                
                print(activation.activating_idx, activation.activation_magnitude)
                print(token_predictions)
                print('-'*20,'\n\n')
                    
            except Exception as e:
                logger.warning(f"Error processing example {i}: {e}")
                assert False
                continue
        
        if not all_predicted_scores:
            logger.warning("No valid predictions obtained")
            return ValidationResult(
                hypothesis=hypothesis,
                correlation=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                false_positive_examples=[],
                false_negative_examples=[]
            )
        
        # Calibrate predictions and calculate correlation
        calibrated_predictions, correlation = self.calibrate_predictions(
            all_predicted_scores,
            all_actual_scores
        )
        
        _, correlation_shifted = self.calibrate_predictions(
            all_predicted_scores,
            all_actual_scores_shifted
        )

        print(f"Correlation (original): {correlation:.3f}", f"scaled: {(correlation+1)/2:.3f}")
        print(f"Correlation (shifted): {correlation_shifted:.3f}", f"scaled: {(correlation_shifted+1)/2:.3f}")
        
        # Calculate precision/recall at percentile threshold
        threshold = np.percentile([s for s in all_actual_scores if s > 0], 50) if any(s > 0 for s in all_actual_scores) else 0.1
        
        true_positives = sum(1 for p, a in zip(calibrated_predictions, all_actual_scores) 
                            if p >= threshold and a >= threshold)
        false_positives = sum(1 for p, a in zip(calibrated_predictions, all_actual_scores) 
                            if p >= threshold and a < threshold)
        false_negatives = sum(1 for p, a in zip(calibrated_predictions, all_actual_scores) 
                            if p < threshold and a >= threshold)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Log some statistics
        logger.info(f"Token-level validation stats:")
        logger.info(f"  Total tokens analyzed: {len(all_predicted_scores)}")
        logger.info(f"  Tokens with actual activation > 0: {sum(1 for a in all_actual_scores if a > 0)}")
        logger.info(f"  Tokens with predicted activation > 0: {sum(1 for p in all_predicted_scores if p > 0)}")
        logger.info(f"  Correlation: {correlation:.3f}")
        
        return ValidationResult(
            hypothesis=hypothesis,
            correlation=correlation,
            precision_at_threshold=precision,
            recall_at_threshold=recall,
            false_positive_examples=[],  # Could add token examples here
            false_negative_examples=[]
        )

    # Update the main validate_hypothesis method to use token-level
    def validate_hypothesis(self, hypothesis: Hypothesis,
                        test_activations: List[LatentActivation],
                        latent_id: int,
                        layer_name: str,
                        use_token_level: bool = True) -> ValidationResult:
        """Validate a hypothesis using either token-level or example-level approach."""
        
        if use_token_level:
            return self.validate_hypothesis_token_level(
                hypothesis, test_activations, latent_id, layer_name
            )
        else:
            # Original example-level validation code
            return self.validate_hypothesis_example_level(
                hypothesis, test_activations, latent_id, layer_name
            )

    def format_example_with_activation_tags(self, example):
        """
        Format the full input with inline activation magnitude tags.
        Shows activations inline as [token|magnitude] for significant activations.
        """
        tokens = example.tokenised_input_text
        
        # Create a mapping of position to activation magnitude
        activation_map = dict(zip(example.activating_idx, example.activation_magnitude))
        
        formatted_text = ""
        for idx, token in enumerate(tokens):
            if idx in activation_map:
                magnitude = activation_map[idx]
                # Tag all activations for this latent (since we're analyzing this specific latent)
                # Use tokenizer to properly convert single tokens
                token_str = self.tokenizer.convert_tokens_to_string([token])
                formatted_text += f" [{token_str.strip()}|{magnitude:.2f}]"
            else:
                formatted_text += self.tokenizer.convert_tokens_to_string([token])
        
        return formatted_text

    
    def extract_activation_contexts(self, 
                                   activations: List[LatentActivation],
                                   dataset: Optional[List] = None) -> List[ActivationContext]:
        """Extract contexts around activations for analysis"""
        contexts = []
        
        # Sort by strongest activation magnitude
        sorted_activations = sorted(
            activations,
            key=lambda x: max(abs(m) for m in x.activation_magnitude),
            reverse=True
        )[:self.top_k_examples]
        
        for activation in sorted_activations:
            # Get the strongest activation position for this example
            max_idx = np.argmax([abs(m) for m in activation.activation_magnitude])
            
            # Extract context
            if activation.tokenised_input_text:
                tokens = activation.tokenised_input_text
            else:
                # Fallback to tokenizing the input text
                if activation.input_text:
                    tokens = self.tokenizer.tokenize(activation.input_text)
                else:
                    logger.warning(f"No text available for example {activation.example_idx}")
                    continue
            
            # Get surrounding context
            act_pos = activation.activating_idx[max_idx]
            start_pos = max(0, act_pos - self.context_window)
            end_pos = min(len(tokens), act_pos + self.context_window + 1)
            
            # Build context strings
            text_before = self.tokenizer.convert_tokens_to_string(tokens[start_pos:act_pos])
            activating_tokens = [activation.activating_token[max_idx]]
            text_after = self.tokenizer.convert_tokens_to_string(tokens[act_pos + 1:end_pos])
            
            contexts.append(ActivationContext(
                text_before=text_before,
                activating_tokens=activating_tokens,
                text_after=text_after,
                activation_strength=activation.activation_magnitude[max_idx],
                token_position=act_pos,
                example_idx=activation.example_idx
            ))
        
        return contexts
    
    
    def generate_hypothesis_prompt(self, contexts: List[ActivationContext]) -> str:
        """Generate prompt for hypothesis generation"""
        prompt = """You are analyzing a latent dimension from a sparse autoencoder trained on a language model. 
I will show you examples where this latent activates strongly, with the activating token(s) marked with <<<>>>.

Your task is to identify the common pattern or concept that this latent represents.

Examples where the latent activates:

"""
        for i, ctx in enumerate(contexts[:10]):  # Show top 10 examples
            prompt += f"Example {i+1} (strength: {abs(ctx.activation_strength):.2f}):\n"
            prompt += f"...{ctx.text_before}<<<{''.join(ctx.activating_tokens)}>>>{ctx.text_after}...\n\n"
        
        prompt += """Based on these examples, what pattern or concept does this latent represent? 
Consider:
1. What linguistic, semantic, or syntactic pattern is common across the activating tokens?
2. Is it responding to specific words, concepts, grammatical structures, or contexts?
3. Be specific and concrete in your hypothesis.

Provide your analysis in the following JSON format:
{
    "hypothesis": "A clear, specific description of what this latent represents",
    "pattern_type": "lexical|semantic|syntactic|contextual",
    "confidence": 0.0-1.0,
    "supporting_observations": ["observation1", "observation2", ...]
}"""
        
        return prompt
    
    def generate_hypothesis(self, 
                           activations: List[LatentActivation],
                           latent_id: int,
                           layer_name: str) -> Hypothesis:
        """Generate a hypothesis about what a latent represents"""
        # Check cache first
        cache_key = f"{layer_name}_{latent_id}_hypothesis.json"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if os.path.exists(cache_path) and False:
            logger.info(f"Loading cached hypothesis for {layer_name} latent {latent_id}")
            with open(cache_path, 'r') as f:
                cached = json.load(f)
                return Hypothesis(**cached)
        
        # Extract contexts
        contexts = self.extract_activation_contexts(activations)
        
        if not contexts:
            logger.warning(f"No valid contexts for {layer_name} latent {latent_id}")
            return Hypothesis(
                description="No activations found",
                confidence=0.0,
                examples_used=[],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        # Generate hypothesis
        # prompt = self.generate_hypothesis_prompt(contexts)
        prompt = self.generate_autointerp_prompt(activations, layer_name, latent_id, max_examples=self.top_k_examples)
        result = self.llm.generate_json(prompt)
        
        hypothesis = Hypothesis(
            description=result.get("hypothesis", "Unknown pattern"),
            confidence=result.get("confidence", 0.5),
            examples_used=[ctx.example_idx for ctx in contexts],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Cache the result
        with open(cache_path, 'w') as f:
            json.dump(asdict(hypothesis), f, indent=2)
        
        return hypothesis
    
    def generate_validation_prompt(self, 
                                  hypothesis: Hypothesis,
                                  test_examples: List[str]) -> str:
        """Generate prompt for validation"""
        prompt = f"""You are simulating a latent dimension that represents: {hypothesis.description}

For each of the following text examples, predict how strongly this latent would activate (0-10 scale):
- 0 means no activation
- 10 means maximum activation
- Consider partial matches (e.g., 3-7)

Analyze each example and predict activation scores based on whether the text contains the pattern described in the hypothesis.

Examples to analyze:

"""
        for i, example in enumerate(test_examples):
            prompt += f"{i+1}. {example}\n"
        
        prompt += """
Provide your predictions in JSON format:
{
    "predictions": [
        {"example_id": 1, "score": 0-10, "reason": "brief explanation"},
        {"example_id": 2, "score": 0-10, "reason": "brief explanation"},
        ...
    ]
}"""
        
        return prompt
    
    def validate_hypothesis_example_level(self,
                           hypothesis: Hypothesis,
                           test_activations: List[LatentActivation],
                           latent_id: int,
                           layer_name: str) -> ValidationResult:
        """Validate a hypothesis on new data"""
        # Extract test examples
        test_contexts = self.extract_activation_contexts(test_activations)
        
        # Prepare test texts
        test_texts = []
        actual_scores = []
        actual_scores_shifted = []
        
        for ctx in test_contexts[:20]:  # Validate on 20 examples
            # Reconstruct a snippet of text around the activation
            text = f"{ctx.text_before} {''.join(ctx.activating_tokens)} {ctx.text_after}"
            test_texts.append(text.strip())
            # Normalize activation strength to 0-10 scale
            actual_scores.append(min(10, abs(ctx.activation_strength)))
            actual_scores_shifted.append(min(10, abs(ctx.activation_strength)))
        
        # Get predictions
        prompt = self.generate_validation_prompt(hypothesis, test_texts)
        result = self.llm.generate_json(prompt)

        
        predictions = result.get("predictions", [])
        if not predictions:
            logger.warning("No predictions returned")
            return ValidationResult(
                hypothesis=hypothesis,
                correlation=0.0,
                precision_at_threshold=0.0,
                recall_at_threshold=0.0,
                false_positive_examples=[],
                false_negative_examples=[]
            )
        
        # Extract predicted scores
        predicted_scores = []
        for pred in predictions:
            predicted_scores.append(pred.get("score", 0))
        
        # Calculate correlation
        if len(predicted_scores) == len(actual_scores):
            print(f"Predicted scores: {predicted_scores}")
            print(f"Actual scores: {actual_scores}")
            scaled_actual_scores = normalize_activations_variance_aware(actual_scores)
            scaled_predicted_scores = normalize_activations_variance_aware(predicted_scores)
            print(f"Scaled predicted scores: {scaled_predicted_scores}")
            print(f"Scaled actual scores: {scaled_actual_scores}")
            correlation, _ = stats.pearsonr(scaled_predicted_scores, scaled_actual_scores)
        else:
            correlation = 0.0
        
        # Calculate precision/recall at threshold
        threshold = 5.0
        true_positives = sum(1 for p, a in zip(predicted_scores, actual_scores) 
                            if p >= threshold and a >= threshold)
        false_positives = sum(1 for p, a in zip(predicted_scores, actual_scores) 
                             if p >= threshold and a < threshold)
        false_negatives = sum(1 for p, a in zip(predicted_scores, actual_scores) 
                             if p < threshold and a >= threshold)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Collect error examples
        false_positive_examples = [
            test_texts[i] for i, (p, a) in enumerate(zip(predicted_scores, actual_scores))
            if p >= threshold and a < threshold
        ][:3]  # Top 3 FPs
        
        false_negative_examples = [
            test_texts[i] for i, (p, a) in enumerate(zip(predicted_scores, actual_scores))
            if p < threshold and a >= threshold
        ][:3]  # Top 3 FNs
        
        return ValidationResult(
            hypothesis=hypothesis,
            correlation=correlation,
            precision_at_threshold=precision,
            recall_at_threshold=recall,
            false_positive_examples=false_positive_examples,
            false_negative_examples=false_negative_examples
        )
    
    def interpret_latent(self,
                        layer_name: str,
                        latent_id: int,
                        hooks_path: str,
                        validation_mode: str,
                        validation_split: float = 0.3) -> Dict:
        """Complete interpretation pipeline for a single latent"""
        logger.info(f"Interpreting {layer_name} latent {latent_id}")
        
        # Load hook data
        hook_file = os.path.join(hooks_path, f"{layer_name}.pkl")
        if not os.path.exists(hook_file):
            raise FileNotFoundError(f"Hook file not found: {hook_file}")
        
        with open(hook_file, 'rb') as f:
            hook_data = pickle.load(f)
        
        # Get activations for the specific latent
        if latent_id not in hook_data.activating_examples:
            raise ValueError(f"Latent {latent_id} not found in {layer_name}")
        
        all_activations = hook_data.activating_examples[latent_id]
        
        if not all_activations:
            logger.warning(f"No activations found for {layer_name} latent {latent_id}")
            return {
                "layer": layer_name,
                "latent": latent_id,
                "status": "dead_neuron",
                "hypothesis": None,
                "validation": None
            }
        
        # Split into train/validation
        n_train = int(len(all_activations) * (1 - validation_split))
        train_activations = all_activations[:n_train]
        val_activations = all_activations[n_train:]
        
        # Generate hypothesis
        # hypothesis = self.generate_hypothesis(train_activations, latent_id, layer_name)
        hypothesis = "This latent encodes concepts related to cold temperature, freezing, refrigeration, and cold food storage/preparation in culinary contexts. It activates on both explicit cold-related terms and references to items that are being frozen, chilled, or refrigerated."
        # logger.info(f"Generated hypothesis: {hypothesis.description}")
        logger.info(f"Generated hypothesis: {hypothesis}")
        
        # Validate hypothesis
        if val_activations:
            # validation = self.validate_hypothesis(hypothesis, val_activations, latent_id, layer_name)
            validation = self.validate_hypothesis(
                    hypothesis, val_activations, latent_id, layer_name,
                    use_token_level=(validation_mode == "token")
                )
            logger.info(f"Validation correlation: {validation.correlation:.3f}")

            logger.info(f"Validation correlation: {validation.correlation:.3f}")
        else:
            validation = None
            logger.warning("No validation data available")
        
        # Compile results
        result = {
            "layer": layer_name,
            "latent": latent_id,
            "status": "active",
            "num_train_activations": len(train_activations),
            "num_val_activations": len(val_activations),
            "hypothesis": asdict(hypothesis),
            "validation": asdict(validation) if validation else None
        }
        
        # Save complete result
        result_path = os.path.join(self.cache_dir, f"{layer_name}_{latent_id}_complete.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result


    def interpret_all_latents(self,
                             layer_name: str,
                             hooks_path: str,
                             validation_mode: str,
                             max_latents: Optional[int] = None,
                             min_activations: int = 5) -> List[Dict]:
        """Interpret all latents in a layer"""
        # Load hook data
        hook_file = os.path.join(hooks_path, f"{layer_name}.pkl")
        if not os.path.exists(hook_file):
            raise FileNotFoundError(f"Hook file not found: {hook_file}")
        
        with open(hook_file, 'rb') as f:
            hook_data = pickle.load(f)
        
        # Get all latent IDs
        all_latent_ids = sorted(hook_data.activating_examples.keys())
        
        # Filter by minimum activations
        active_latents = [
            lid for lid in all_latent_ids 
            if len(hook_data.activating_examples[lid]) >= min_activations
        ]
        
        logger.info(f"Found {len(active_latents)} active latents (out of {len(all_latent_ids)} total)")
        
        # Limit if requested
        if max_latents:
            active_latents = active_latents[:max_latents]
        
        results = []
        
        # Process each latent
        for latent_id in tqdm(active_latents, desc=f"Interpreting {layer_name} latents"):
            try:
                result = self.interpret_latent(layer_name, latent_id, hooks_path, validation_mode)
                results.append(result)
                
                # Save intermediate results
                if len(results) % 10 == 0:
                    summary_path = os.path.join(self.cache_dir, f"{layer_name}_summary_partial.json")
                    with open(summary_path, 'w') as f:
                        json.dump(results, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error interpreting latent {latent_id}: {e}")
                results.append({
                    "layer": layer_name,
                    "latent": latent_id,
                    "status": "error",
                    "error": str(e)
                })
        
        # Save final summary
        summary_path = os.path.join(self.cache_dir, f"{layer_name}_summary_complete.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


def analyze_results_summary(results: List[Dict]) -> Dict:
    """Generate summary statistics from interpretation results"""
    summary = {
        "total_latents": len(results),
        "active_latents": sum(1 for r in results if r.get("status") == "active"),
        "dead_latents": sum(1 for r in results if r.get("status") == "dead_neuron"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "avg_correlation": 0.0,
        "avg_precision": 0.0,
        "avg_recall": 0.0,
        "high_quality_latents": 0,
        "interpretable_latents": 0
    }
    
    # Calculate averages for validated latents
    validated_results = [
        r for r in results 
        if r.get("validation") and r["validation"].get("correlation") is not None
    ]
    
    if validated_results:
        correlations = [r["validation"]["correlation"] for r in validated_results]
        precisions = [r["validation"]["precision_at_threshold"] for r in validated_results]
        recalls = [r["validation"]["recall_at_threshold"] for r in validated_results]
        
        summary["avg_correlation"] = np.mean(correlations)
        summary["avg_precision"] = np.mean(precisions)
        summary["avg_recall"] = np.mean(recalls)
        
        # Count high quality interpretations (correlation > 0.5)
        summary["high_quality_latents"] = sum(1 for c in correlations if c > 0.5)
        
        # Count interpretable latents (correlation > 0.3)
        summary["interpretable_latents"] = sum(1 for c in correlations if c > 0.3)
    
    return summary

def normalize_activations_percentile(activations: List[float], 
                                   reference_activations: Optional[List[float]] = None) -> List[float]:
    """
    Normalize activations to 0-10 scale using percentile mapping.
    
    Args:
        activations: List of activation magnitudes to normalize
        reference_activations: Optional reference distribution (e.g., from training data)
                             If None, uses the activations themselves
    """
    if reference_activations is None:
        reference_activations = activations
    
    # Use absolute values
    abs_activations = [abs(a) for a in activations]
    abs_reference = [abs(a) for a in reference_activations]
    
    # Calculate percentiles
    normalized = []
    for act in abs_activations:
        percentile = (sum(1 for ref in abs_reference if ref <= act) / len(abs_reference)) * 100
        # Map percentile (0-100) to score (0-10)
        score = percentile / 10
        normalized.append(score)
    
    return normalized

def normalize_activations_variance_aware(activations: List[float]) -> List[float]:
    """
    Normalize activations considering their variance.
    Uses different strategies for high vs low variance data.
    """
    abs_activations = [abs(a) for a in activations]
    
    # Calculate statistics
    mean = np.mean(abs_activations)
    std = np.std(abs_activations)
    cv = std / mean if mean > 0 else 0  # Coefficient of variation
    
    # If variance is low (CV < 0.1), use absolute scale
    if cv < 0.1:
        # Use absolute magnitude thresholds
        normalized = []
        for act in abs_activations:
            if act < 1:
                score = act * 2  # 0-1 maps to 0-2
            elif act < 3:
                score = 2 + (act - 1)  # 1-3 maps to 2-4
            elif act < 5:
                score = 4 + (act - 3) * 1.5  # 3-5 maps to 4-7
            elif act < 7:
                score = 7 + (act - 5) * 1.5  # 5-7 maps to 7-10
            else:
                score = 10  # >7 maps to 10
            normalized.append(min(10, score))
    else:
        # High variance: use percentile-based
        sorted_acts = sorted(abs_activations)
        normalized = []
        for act in abs_activations:
            percentile = (sum(1 for a in sorted_acts if a <= act) / len(sorted_acts)) * 100
            score = percentile / 10
            normalized.append(score)
    
    return normalized


def main():
    # print(normalize_activations_variance_aware([6.8853559494018555, 6.800094127655029, 6.72633171081543, 6.475478649139404, 6.427580833435059, 6.305151462554932, 6.302254676818848, 6.299530982971191, 6.292999744415283, 6.290709972381592, 6.260174751281738, 6.242665767669678, 6.229837894439697, 6.225137233734131, 6.223772048950195, 6.221789360046387, 6.215948581695557, 6.215138912200928, 6.2144455909729, 6.206278324127197]))
    # assert False
    parser = argparse.ArgumentParser(description="Interpret LoRA latents using LLMs")
    parser.add_argument("--layer", type=str, required=True, 
                       help="Layer name (e.g., 'model.layers.11.mlp.down_proj')")
    parser.add_argument("--latent", type=int, default=None,
                       help="Specific latent ID to analyze (e.g., 733). If not provided, analyzes all latents.")
    parser.add_argument("--hooks-path", type=str, required=True,
                       help="Path to directory containing hook pickle files")
    parser.add_argument("--max-latents", type=int, default=None,
                       help="Maximum number of latents to analyze (for testing)")
    parser.add_argument("--min-activations", type=int, default=5,
                       help="Minimum number of activations required to analyze a latent")
    parser.add_argument("--llm", type=str, choices=["openai", "vllm"], default="vllm",
                       help="LLM provider to use")
    parser.add_argument("--openai-key", type=str, default=None,
                       help="OpenAI API key (if using OpenAI)")
    parser.add_argument("--openai-model", type=str, default="gpt-4-turbo-preview",
                       help="OpenAI model to use")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000",
                       help="vLLM server URL")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ",
                       help="vLLM model name")
    parser.add_argument("--tokenizer", type=str, default="/home/cvenhoff/lora_interp/experiments/gemma-2-2b_topk_dpo_r1024_k8_steps5000/final_adapter",
                       help="Tokenizer to use")
    parser.add_argument("--context-window", type=int, default=20,
                       help="Context window size around activations")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Number of top activations to use")
    parser.add_argument("--cache-dir", type=str, default="cache/interpretations",
                       help="Directory for caching results")
    parser.add_argument("--validation-mode", type=str, choices=["token", "example"], 
                   default="token", help="Validation mode: token-level or example-level")

    
    args = parser.parse_args()
    
    # Initialize LLM interface
    if args.llm == "openai":
        if os.environ.get("OPENAI_API_KEY", None) is None and args.openai_key is None:
            raise ValueError("OpenAI API key required when using OpenAI")
        llm = OpenAIInterface(os.environ.get("OPENAI_API_KEY"), args.openai_model)
    else:
        llm = VLLMInterface(args.vllm_model, args.vllm_url)
    
    # Initialize interpreter
    interpreter = LatentInterpreter(
        llm=llm,
        tokenizer_name=args.tokenizer,
        context_window=args.context_window,
        top_k_examples=args.top_k,
        cache_dir=args.cache_dir
    )
    
    # Run interpretation
    try:
        if args.latent is not None:
            # Single latent mode
            result = interpreter.interpret_latent(
                layer_name=args.layer,
                latent_id=args.latent,
                hooks_path=args.hooks_path,
                validation_mode=args.validation_mode,
            )
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"Interpretation Results for {args.layer} Latent {args.latent}")
            print(f"{'='*60}")
            print(f"Status: {result['status']}")
            
            if result['hypothesis']:
                print(f"\nHypothesis: {result['hypothesis']['description']}")
                print(f"Confidence: {result['hypothesis']['confidence']:.2f}")
            
            if result['validation']:
                print(f"\nValidation Results:")
                print(f"  Correlation: {result['validation']['correlation']:.3f}")
                print(f"  Precision: {result['validation']['precision_at_threshold']:.3f}")
                print(f"  Recall: {result['validation']['recall_at_threshold']:.3f}")
                
                if result['validation']['false_positive_examples']:
                    print(f"\nFalse Positive Examples:")
                    for ex in result['validation']['false_positive_examples']:
                        print(f"  - {ex[:100]}...")
            
            print(f"\nFull results saved to: {args.cache_dir}")
            
        else:
            # All latents mode
            logger.info(f"Analyzing all latents in {args.layer}")
            results = interpreter.interpret_all_latents(
                layer_name=args.layer,
                hooks_path=args.hooks_path,
                max_latents=args.max_latents,
                min_activations=args.min_activations,
                validation_mode=args.validation_mode
            )
            
            # Generate and print summary
            summary = analyze_results_summary(results)
            
            print(f"\n{'='*60}")
            print(f"Summary for {args.layer} ({summary['total_latents']} latents)")
            print(f"{'='*60}")
            print(f"Active latents: {summary['active_latents']}")
            print(f"Dead latents: {summary['dead_latents']}")
            print(f"Errors: {summary['errors']}")
            
            if summary['active_latents'] > 0:
                print(f"\nValidation Metrics (averaged):")
                print(f"  Correlation: {summary['avg_correlation']:.3f}")
                print(f"  Precision: {summary['avg_precision']:.3f}")
                print(f"  Recall: {summary['avg_recall']:.3f}")
                print(f"\nQuality Distribution:")
                print(f"  High quality (corr > 0.5): {summary['high_quality_latents']}")
                print(f"  Interpretable (corr > 0.3): {summary['interpretable_latents']}")
            
            # Show top interpretations
            if results:
                sorted_results = sorted(
                    [r for r in results if r.get('validation') and r['validation'].get('correlation')],
                    key=lambda x: x['validation']['correlation'],
                    reverse=True
                )[:5]
                
                if sorted_results:
                    print(f"\nTop 5 Best Interpreted Latents:")
                    for i, r in enumerate(sorted_results):
                        print(f"\n{i+1}. Latent {r['latent']} (correlation: {r['validation']['correlation']:.3f})")
                        print(f"   Hypothesis: {r['hypothesis']['description']}")
            
            print(f"\nFull results saved to: {os.path.join(args.cache_dir, args.layer + '_summary_complete.json')}")
        
    except Exception as e:
        logger.error(f"Error during interpretation: {e}")
        raise


if __name__ == "__main__":
    main()


# result = {'predictions': [{'example_id': 1, 'score': 8, 'reason': "The text contains multiple instances of the copula 'to be' attached to personal pronouns ('I'm', 'you’re', 'you are')."}, {'example_id': 2, 'score': 3, 'reason': "The text contains fewer instances but still has 'I'm' which matches the criteria."}, {'example_id': 3, 'score': 2, 'reason': "One instance of 'I was', indicating a past form of the copula 'to be' with personal pronoun."}, {'example_id': 4, 'score': 2, 'reason': "Identical to example 3, containing 'I was'."}, {'example_id': 5, 'score': 7, 'reason': "Contains multiple instances ('I’m', 'you’re'), clearly matching the pattern of interest."}, {'example_id': 6, 'score': 2, 'reason': "Repeats the instance 'I was' from example 3."}, {'example_id': 7, 'score': 6, 'reason': "Contains 'I'm' and 'I am', directly referring to the pattern."}, {'example_id': 8, 'score': 5, 'reason': "Contains 'I am' twice, fitting the pattern but in a slightly less frequent manner."}, {'example_id': 9, 'score': 0, 'reason': "Does not contain instances of the copula 'to be' attached to personal pronouns as described."}, {'example_id': 10, 'score': 4, 'reason': "Contains 'I'm', matching the pattern but not as frequently."}, {'example_id': 11, 'score': 2, 'reason': "Contains 'I was', indicating the past form of the copula 'to be' with a personal pronoun."}, {'example_id': 12, 'score': 6, 'reason': "Contains 'you’re', matching the pattern of interest."}, {'example_id': 13, 'score': 5, 'reason': "Contains 'you’re', fitting the pattern described."}, {'example_id': 14, 'score': 3, 'reason': "Contains 'It’s' and 'they’re', which are forms of the copula 'to be' but less directly attached to personal pronouns."}, {'example_id': 15, 'score': 1, 'reason': "Contains 'be', but not directly attached to a personal pronoun in the given context."}, {'example_id': 16, 'score': 5, 'reason': "Contains 'I’m' and 'you’re', directly matching the pattern of interest."}, {'example_id': 17, 'score': 7, 'reason': "Contains 'I’m' and 'you’re', fitting the pattern strongly."}, {'example_id': 18, 'score': 0, 'reason': "Does not contain instances of the copula 'to be' attached to personal pronouns."}, {'example_id': 19, 'score': 2, 'reason': "Contains 'is', which is a form of the copula 'to be', but not attached to a personal pronoun as described."}, {'example_id': 20, 'score': 4, 'reason': "Contains 'I am', which fits the pattern described."}]}
# # print(f"Validation prompt sent to LLM:\n{prompt}\n")
# # print(f"LLM response:\n{result}\n")
# # assert False