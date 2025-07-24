#!/usr/bin/env python3
"""
LoRA Latent Interpreter - Generates and validates hypotheses about sparse LoRA adapter latents
Modified to support OpenAI Batch API for cost-effective processing
"""

import argparse
import json
import pickle
import os
import re
import time
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Union, Any, Any
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

class OpenAIBatchInterface(LLMInterface):
    """OpenAI API interface with batch support"""
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview", use_batch: bool = True):
        import openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.use_batch = use_batch
        self.batch_requests = []
    
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
        if self.use_batch:
            # Store request for later batch processing
            request_id = str(uuid.uuid4())
            self.batch_requests.append(BatchRequest(
                request_id=request_id,
                request_type="generate",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            ))
            return request_id  # Return request ID as placeholder
        else:
            # Immediate processing
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
    
    def generate_json(self, prompt: str, max_tokens: int = 8000) -> Union[Dict, str]:
        if self.use_batch:
            # Store request for later batch processing
            request_id = str(uuid.uuid4())
            self.batch_requests.append(BatchRequest(
                request_id=request_id,
                request_type="generate_json",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=1.0,
                response_format={"type": "json_object"}
            ))
            return request_id  # Return request ID as placeholder
        else:
            # Immediate processing
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
                temperature=1.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
    
    def generate_batch(self, requests: List[BatchRequest]) -> Dict[str, Any]:
        """Process batch requests using OpenAI Batch API"""
        if not requests:
            return {}
        
        # Create JSONL content for batch
        batch_data = []
        for req in requests:
            messages = [{"role": "user", "content": req.prompt}]
            
            batch_item = {
                "custom_id": req.request_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": messages,
                    "max_completion_tokens": req.max_tokens,
                    "temperature": req.temperature
                }
            }
            
            if req.response_format:
                batch_item["body"]["response_format"] = req.response_format
            
            batch_data.append(batch_item)
        
        # Save batch file
        batch_filename = f"batch_requests_{int(time.time())}.jsonl"
        with open(batch_filename, 'w') as f:
            for item in batch_data:
                f.write(json.dumps(item) + '\n')
        
        try:
            # Upload batch file
            logger.info(f"Uploading batch file with {len(requests)} requests...")
            with open(batch_filename, 'rb') as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose='batch'
                )
            
            # Create batch job
            batch_job = self.client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            logger.info(f"Batch job created: {batch_job.id}")
            logger.info("Waiting for batch completion...")
            
            # Poll for completion
            while True:
                batch_status = self.client.batches.retrieve(batch_job.id)
                logger.info(f"Batch status: {batch_status.status}")
                
                if batch_status.status == "completed":
                    break
                elif batch_status.status in ["failed", "expired", "cancelled"]:
                    raise Exception(f"Batch failed with status: {batch_status.status}")
                
                time.sleep(30)  # Wait 30 seconds before checking again
            
            # Download results
            result_file_id = batch_status.output_file_id
            result_content = self.client.files.content(result_file_id)
            
            # Parse results
            results = {}
            for line in result_content.text.strip().split('\n'):
                result_item = json.loads(line)
                custom_id = result_item['custom_id']
                
                if result_item.get('response') and result_item['response'].get('body'):
                    response_body = result_item['response']['body']
                    if response_body.get('choices'):
                        content = response_body['choices'][0]['message']['content']
                        results[custom_id] = content
                    else:
                        logger.warning(f"No choices in response for {custom_id}")
                        results[custom_id] = None
                else:
                    logger.warning(f"Error in batch result for {custom_id}: {result_item.get('error', 'Unknown error')}")
                    results[custom_id] = None
            
            logger.info(f"Batch completed successfully. Got {len(results)} results.")
            return results
            
        finally:
            # Clean up batch file
            if os.path.exists(batch_filename):
                os.remove(batch_filename)
    
    def process_accumulated_requests(self) -> Dict[str, Any]:
        """Process all accumulated batch requests"""
        if not self.batch_requests:
            return {}
        
        logger.info(f"Processing {len(self.batch_requests)} accumulated requests...")
        results = self.generate_batch(self.batch_requests)
        self.batch_requests = []  # Clear after processing
        return results

class VLLMInterface(LLMInterface):
    """vLLM interface for local models (unchanged for compatibility)"""
    
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
    """Modified LatentInterpreter with batch processing support"""
    
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

    # [Include all the existing methods from LatentInterpreter here, but modify the ones that call LLM]
    
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

    def parse_token_predictions(self, response_content: str, num_tokens: int) -> Dict[int, float]:
        """Parse token-level predictions from response content."""
        predictions = {i: 0.0 for i in range(num_tokens)}  # Initialize all to 0
        
        try:
            # Handle both dict and string responses
            if isinstance(response_content, str):
                response = json.loads(response_content)
            else:
                response = response_content
                
            if "predictions" in response:
                for pred in response["predictions"]:
                    pos = pred.get("position", -1)
                    activation = pred.get("activation", 0.0)
                    
                    if 0 <= pos < num_tokens:
                        # Ensure activation is in [0, 1] range
                        predictions[pos] = max(0.0, min(1.0, float(activation)))
                        
        except Exception as e:
            logger.warning(f"Error parsing token predictions: {e}")
        
        return predictions

    def generate_autointerp_prompt(self, activation_examples, module_name, latent_idx, max_examples=10):
        """Generate a comprehensive interpretability prompt for a specific TopK LoRA latent."""
                 
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

    def format_example_with_activation_tags(self, example):
        """Format the full input with inline activation magnitude tags."""
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

    def generate_hypothesis_batch(self, 
                                activations_list: List[Tuple[List[LatentActivation], int, str]]) -> List[str]:
        """Generate hypotheses for multiple latents in batch mode"""
        request_ids = []
        
        for activations, latent_id, layer_name in activations_list:
            prompt = self.generate_autointerp_prompt(activations, layer_name, latent_id)
            
            if hasattr(self.llm, 'use_batch') and self.llm.use_batch:
                request_id = self.llm.generate_json(prompt)  # This returns request_id in batch mode
                request_ids.append(request_id)
                # Store context for later use
                self.pending_requests[request_id] = {
                    'type': 'hypothesis',
                    'latent_id': latent_id,
                    'layer_name': layer_name,
                    'activations': activations
                }
            else:
                # Immediate processing for non-batch mode
                result = self.llm.generate_json(prompt)
                request_ids.append(result)
        
        return request_ids

    def validate_hypothesis_token_level_batch(self, 
                                           hypothesis_validation_pairs: List[Tuple[Hypothesis, List[LatentActivation], int, str]]) -> List[str]:
        """Validate multiple hypotheses in batch mode"""
        request_ids = []
        
        for hypothesis, test_activations, latent_id, layer_name in hypothesis_validation_pairs:
            # Process first few examples for token-level validation
            for i, activation in enumerate(test_activations[:5]):  # Max 5 examples per hypothesis
                prompt = self.create_token_simulation_prompt(
                    hypothesis.description,
                    activation
                )
                
                if hasattr(self.llm, 'use_batch') and self.llm.use_batch:
                    request_id = self.llm.generate_json(prompt, max_tokens=8000)
                    request_ids.append(request_id)
                    # Store context for later use
                    self.pending_requests[request_id] = {
                        'type': 'validation',
                        'hypothesis': hypothesis,
                        'activation': activation,
                        'latent_id': latent_id,
                        'layer_name': layer_name,
                        'example_idx': i
                    }
                else:
                    # Immediate processing
                    result = self.llm.generate_json(prompt, max_tokens=8000)
                    request_ids.append(result)
        
        return request_ids

    def process_batch_results(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch results and organize them by original context"""
        processed_results = {
            'hypotheses': {},
            'validations': {}
        }
        
        for request_id, response_content in batch_results.items():
            if request_id not in self.pending_requests:
                logger.warning(f"Unknown request ID: {request_id}")
                continue
            
            context = self.pending_requests[request_id]
            
            if context['type'] == 'hypothesis':
                try:
                    if isinstance(response_content, str):
                        result = json.loads(response_content)
                    else:
                        result = response_content
                    
                    hypothesis = Hypothesis(
                        description=result.get("hypothesis", "Unknown pattern"),
                        confidence=result.get("confidence", 0.5),
                        examples_used=[],  # Will be filled later
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                    )
                    
                    key = f"{context['layer_name']}_{context['latent_id']}"
                    processed_results['hypotheses'][key] = hypothesis
                    
                except Exception as e:
                    logger.error(f"Error processing hypothesis result for {request_id}: {e}")
            
            elif context['type'] == 'validation':
                try:
                    if isinstance(response_content, str):
                        result = json.loads(response_content)
                    else:
                        result = response_content
                    
                    token_predictions = self.parse_token_predictions(
                        result, 
                        len(context['activation'].tokenised_input_text)
                    )
                    
                    key = f"{context['layer_name']}_{context['latent_id']}"
                    if key not in processed_results['validations']:
                        processed_results['validations'][key] = []
                    
                    processed_results['validations'][key].append({
                        'predictions': token_predictions,
                        'activation': context['activation'],
                        'example_idx': context['example_idx']
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing validation result for {request_id}: {e}")
        
        # Clear pending requests
        self.pending_requests = {}
        return processed_results

    def interpret_latents_batch(self,
                              layer_name: str,
                              hooks_path: str,
                              validation_mode: str = "token",
                              max_latents: Optional[int] = None,
                              min_activations: int = 5,
                              validation_split: float = 0.3) -> List[Dict]:
        """Interpret multiple latents using batch processing"""
        logger.info(f"Starting batch interpretation for {layer_name}")
        
        # Load hook data
        hook_file = os.path.join(hooks_path, f"{layer_name}.pkl")
        if not os.path.exists(hook_file):
            raise FileNotFoundError(f"Hook file not found: {hook_file}")
        
        with open(hook_file, 'rb') as f:
            hook_data = pickle.load(f)
        
        # Get active latents
        all_latent_ids = sorted(hook_data.activating_examples.keys())
        active_latents = [
            lid for lid in all_latent_ids 
            if len(hook_data.activating_examples[lid]) >= min_activations
        ]
        
        if max_latents:
            active_latents = active_latents[:max_latents]
        
        logger.info(f"Processing {len(active_latents)} latents in batch mode")
        
        # Prepare data for batch processing
        hypothesis_data = []
        validation_data = []
        
        for latent_id in active_latents:
            all_activations = hook_data.activating_examples[latent_id]
            n_train = int(len(all_activations) * (1 - validation_split))
            train_activations = all_activations[:n_train]
            val_activations = all_activations[n_train:]
            
            # Add to hypothesis batch
            hypothesis_data.append((train_activations, latent_id, layer_name))
            
            # Add to validation batch (we'll generate hypotheses first)
            if val_activations:
                validation_data.append((None, val_activations, latent_id, layer_name))  # Hypothesis will be filled later
        
        # Step 1: Generate all hypotheses in batch
        logger.info("Generating hypotheses in batch...")
        hypothesis_request_ids = self.generate_hypothesis_batch(hypothesis_data)
        
        # Process batch if using OpenAI batch mode
        if hasattr(self.llm, 'process_accumulated_requests'):
            batch_results = self.llm.process_accumulated_requests()
            hypothesis_results = self.process_batch_results(batch_results)
        else:
            # For immediate processing modes
            hypothesis_results = {'hypotheses': {}, 'validations': {}}
            for i, (train_activations, latent_id, layer_name) in enumerate(hypothesis_data):
                result = hypothesis_request_ids[i]  # This would be the actual result
                key = f"{layer_name}_{latent_id}"
                # Process result...
        
        # Step 2: Generate all validations in batch
        logger.info("Generating validations in batch...")
        validation_batch_data = []
        for _, val_activations, latent_id, layer_name in validation_data:
            key = f"{layer_name}_{latent_id}"
            if key in hypothesis_results['hypotheses']:
                hypothesis = hypothesis_results['hypotheses'][key]
                validation_batch_data.append((hypothesis, val_activations, latent_id, layer_name))
        
        validation_request_ids = self.validate_hypothesis_token_level_batch(validation_batch_data)
        
        # Process validation batch
        if hasattr(self.llm, 'process_accumulated_requests'):
            batch_results = self.llm.process_accumulated_requests()
            validation_results = self.process_batch_results(batch_results)
        
        # Step 3: Combine results
        final_results = []
        for latent_id in active_latents:
            key = f"{layer_name}_{latent_id}"
            
            result = {
                "layer": layer_name,
                "latent": latent_id,
                "status": "active"
            }
            
            if key in hypothesis_results['hypotheses']:
                result['hypothesis'] = asdict(hypothesis_results['hypotheses'][key])
            
            if key in validation_results.get('validations', {}):
                # Process validation results into ValidationResult
                val_data = validation_results['validations'][key]
                # Calculate correlation and other metrics
                # ... (implement correlation calculation from token predictions)
                result['validation'] = {
                    'correlation': 0.0,  # Calculate from val_data
                    'precision_at_threshold': 0.0,
                    'recall_at_threshold': 0.0,
                    'false_positive_examples': [],
                    'false_negative_examples': []
                }
            
            final_results.append(result)
        
        return final_results

# Update main function to support batch mode
def main():
    parser = argparse.ArgumentParser(description="Interpret LoRA latents using LLMs with batch support")
    parser.add_argument("--layer", type=str, required=True, 
                       help="Layer name (e.g., 'model.layers.11.mlp.down_proj')")
    parser.add_argument("--latent", type=int, default=None,
                       help="Specific latent ID to analyze. If not provided, analyzes all latents.")
    parser.add_argument("--hooks-path", type=str, required=True,
                       help="Path to directory containing hook pickle files")
    parser.add_argument("--max-latents", type=int, default=None,
                       help="Maximum number of latents to analyze")
    parser.add_argument("--min-activations", type=int, default=5,
                       help="Minimum number of activations required")
    parser.add_argument("--llm", type=str, choices=["openai", "vllm"], default="openai",
                       help="LLM provider to use")
    parser.add_argument("--use-batch", action="store_true", default=True,
                       help="Use batch processing for OpenAI (more cost-effective)")
    parser.add_argument("--openai-key", type=str, default=None,
                       help="OpenAI API key")
    parser.add_argument("--openai-model", type=str, default="gpt-4-turbo-preview",
                       help="OpenAI model to use")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000",
                       help="vLLM server URL")
    parser.add_argument("--vllm-model", type=str, default="Qwen/Qwen2.5-32B-Instruct-AWQ",
                       help="vLLM model name")
    parser.add_argument("--tokenizer", type=str, default="google/gemma-2-2b",
                       help="Tokenizer to use")
    parser.add_argument("--cache-dir", type=str, default="cache/interpretations_batch",
                       help="Directory for caching results")
    parser.add_argument("--validation-mode", type=str, choices=["token", "example"], 
                       default="token", help="Validation mode")
    
    args = parser.parse_args()
    
    # Initialize LLM interface
    if args.llm == "openai":
        api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required")
        llm = OpenAIBatchInterface(api_key, args.openai_model, use_batch=args.use_batch)
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
    if args.latent is not None:
        logger.info("Single latent mode - batch processing not applicable")
        # Fall back to individual processing
        # ... implement single latent processing
    else:
        # Batch processing mode
        results = interpreter.interpret_latents_batch(
            layer_name=args.layer,
            hooks_path=args.hooks_path,
            validation_mode=args.validation_mode,
            max_latents=args.max_latents,
            min_activations=args.min_activations
        )
        
        logger.info(f"Completed batch processing of {len(results)} latents")
        
        # Save results
        output_file = os.path.join(args.cache_dir, f"{args.layer}_batch_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

"""
Usage Examples:

# Sequential processing (original behavior)
python lora_interpreter_batch.py --layer "model.layers.11.mlp.down_proj" --hooks-path /path/to/hooks --llm openai

# Batch processing (50% cost reduction with OpenAI)
python lora_interpreter_batch.py --layer "model.layers.11.mlp.down_proj" --hooks-path /path/to/hooks --llm openai --use-batch

# Process limited number of latents in batch mode
python lora_interpreter_batch.py --layer "model.layers.11.mlp.down_proj" --hooks-path /path/to/hooks --llm openai --use-batch --max-latents 50

Note: Batch processing requires patience as OpenAI batches can take up to 24 hours to complete,
but typically finish much faster. The cost savings (50%) make it worthwhile for large-scale analysis.
"""