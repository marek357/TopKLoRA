import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import gc

# For efficient inference
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('interpretability_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def format_example_with_activation_tags(example, tokenizer):
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
            token_str = tokenizer.convert_tokens_to_string([token])
            formatted_text += f"[{token_str.strip()}|{magnitude:.2f}]"
        else:
            formatted_text += tokenizer.convert_tokens_to_string([token])
    
    return formatted_text


def generate_autointerp_prompt(latent_data, module_name, latent_idx, tokenizer, max_examples=10):
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
    
    # Filter to only examples with causal effects
    examples_with_effects = [ex for ex in latent_data['examples'] if ex.ablation_has_effect]
    
    # Handle case where latent has no causal effects
    if not examples_with_effects:
        return f"""## Non-Causal Latent Detected

You are analyzing latent {latent_idx} in module {module_name} from a TopK LoRA adapter.

**Key Finding**: This latent activated {latent_data['num_activations']} times across the dataset, but ablating it NEVER changed the model's output.

This suggests one of the following:
1. The latent is redundant (other latents compensate for its function)
2. The latent only matters in combination with other latents
3. The latent encodes information that doesn't affect the model's generation behavior
4. The latent may be undertrained or "dead"

No further interpretation is needed for this non-causal latent."""
    
    # Build prompt for causal latents
    prompt = f"""You are analyzing a latent component in a TopK LoRA adapter from a language model. This latent is part of a low-rank decomposition where only the top-k most active components are used during each forward pass.

## Latent Information
- Module: {module_name}
- Latent Index: {latent_idx}
- Total Activations Found: {latent_data['num_activations']}
- Causal Examples: {len(examples_with_effects)} (out of {len(latent_data['examples'])} tested)
- Causal Effect Rate: {latent_data['effect_rate']:.1%} (percentage of tested examples where ablating this latent changed the output)

## Important Context
This analysis only shows examples where ablating (disabling) this latent actually changed the model's output. These causal examples are most informative for understanding what functional role this latent plays in the model's computation.

## Causal Examples

Below are examples where this latent both activated strongly AND causally influenced the model's output when disabled. For each example:
- **Full input** is shown with activation magnitudes inline as [token|magnitude]
- **Output comparison** shows how the model's generation changed when this latent was disabled

"""
    
    # Sort examples by activation magnitude
    examples = sorted(examples_with_effects[:max_examples], 
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
        
        # Show full input with activation tags
        prompt += f"\n**Full Input (with activation magnitudes):**\n```\n"
        prompt += format_example_with_activation_tags(example, tokenizer)
        prompt += f"\n```\n"
        
        # Show how ablation changed the output
        prompt += f"\n**Output Change When Latent Disabled:**\n"
        
        # Find where outputs diverge
        baseline_words = example.baseline_text.split()
        ablated_words = example.ablated_text.split()
        
        # Find divergence point
        diverge_idx = len(baseline_words)  # Default to end if they're completely different
        for j, (b, a) in enumerate(zip(baseline_words, ablated_words)):
            if b != a:
                diverge_idx = j
                break
        
        # Show more context around divergence
        if diverge_idx < len(baseline_words):
            # Show 10 words before and 20 after divergence
            context_start = max(0, diverge_idx - 10)
            context_end = min(max(len(baseline_words), len(ablated_words)), diverge_idx + 20)
            
            baseline_context = " ".join(baseline_words[context_start:context_end])
            ablated_context = " ".join(ablated_words[context_start:context_end] if context_start < len(ablated_words) else ["[GENERATION ENDED]"])
            
            prompt += f"- **Original continuation**: ...{baseline_context}...\n"
            prompt += f"- **With latent disabled**: ...{ablated_context}...\n"
            
            # Note if one is significantly shorter
            if len(ablated_words) < len(baseline_words) - 5:
                prompt += f"  *(Note: Ablated version is {len(baseline_words) - len(ablated_words)} words shorter)*\n"
            elif len(ablated_words) > len(baseline_words) + 5:
                prompt += f"  *(Note: Ablated version is {len(ablated_words) - len(baseline_words)} words longer)*\n"
        else:
            # Complete mismatch from the start
            prompt += f"- **Original**: {' '.join(baseline_words[:30])}...\n"
            prompt += f"- **Ablated**: {' '.join(ablated_words[:30])}...\n"
            prompt += f"  *(Note: Outputs diverge immediately)*\n"
        
        prompt += "\n---\n"
    
    # Add detailed interpretation instructions
    prompt += """
## Interpretation Task

Based on the causal examples above, form a hypothesis about what this latent represents and how it functions in the model. Consider the following aspects:

### 1. **Input Pattern Analysis**
- What do the tokens with high activation magnitudes have in common?
- Are there semantic patterns (meaning-based), syntactic patterns (grammar-based), or pragmatic patterns (context/discourse-based)?
- Do activations cluster around specific parts of the input (e.g., certain types of phrases, questions, statements)?
- Are activation magnitudes consistent across similar tokens, or do they vary based on context?

### 2. **Functional Role Analysis**
- How does disabling this latent change the model's behavior?
- Does it affect the style, content, safety, factuality, or other aspects of generation?
- Are the changes consistent across examples (e.g., always making text more verbose, always removing certain types of content)?
- Does the latent seem to enable or suppress certain behaviors?

### 3. **Contextual Patterns**
- In what types of conversations or text does this latent activate?
- Does it respond to specific topics, interaction patterns, or discourse structures?
- Are there common scenarios across the examples?

### 4. **Magnitude Patterns**
- Do certain contexts trigger stronger activations?
- Is there a relationship between activation strength and the size of the effect when ablated?
- Are negative and positive activations associated with different functions?

## Required Output Format

Please provide your analysis in the following structured format:

**Hypothesis**: [A clear, concise 1-2 sentence description of what this latent encodes or detects, and its functional role]

**Evidence**: 
- [Key observation 1 with specific example references]
- [Key observation 2 with specific example references]
- [Key observation 3 with specific example references]
- [Additional observations as needed]

**Pattern Summary**:
- Input patterns: [What types of tokens/contexts activate this latent]
- Output effects: [How the model's behavior changes when this latent is disabled]
- Activation characteristics: [Notable patterns in magnitude, sign, or distribution]

**Confidence**: [low/medium/high] - with brief justification

**Alternative Interpretations**: [Any other plausible interpretations you considered and why they're less likely]

**Uncertainties**: [Aspects you couldn't fully explain or would need more examples to understand]

## Important Notes
- Focus on patterns that appear across multiple examples
- Be specific about which examples support each part of your interpretation
- If activation magnitude patterns suggest different "modes" of the latent, describe them
- Consider both what the latent detects (input side) and what it controls (output side)
"""
    
    return prompt


# Example usage function
def run_autointerp_analysis(ablation_results, module_name, latent_idx, tokenizer):
    """
    Run the autointerp analysis for a specific latent.
    
    Args:
        ablation_results: The full results dictionary from your analysis
        module_name: Name of the module to analyze
        latent_idx: Index of the latent to interpret
        tokenizer: The tokenizer used by the model
    """
    
    # Extract latent data
    latent_data = ablation_results[module_name][latent_idx]
    
    # Generate the prompt
    prompt = generate_autointerp_prompt(
        latent_data=latent_data,
        module_name=module_name,
        latent_idx=latent_idx,
        tokenizer=tokenizer,
        max_examples=10
    )
    
    return prompt


# Optional: Batch processing function for multiple latents
def generate_batch_autointerp_prompts(ablation_results, module_name, tokenizer, 
                                     top_n_latents=20, min_effect_rate=0.1):
    """
    Generate prompts for the top N most impactful latents in a module.
    
    Args:
        ablation_results: The full results dictionary
        module_name: Name of the module to analyze  
        tokenizer: The tokenizer
        top_n_latents: Number of top latents to analyze
        min_effect_rate: Minimum effect rate to consider a latent worth analyzing
    """
    
    module_results = ablation_results[module_name]
    
    # Filter and sort latents by effect rate
    causal_latents = [
        (idx, data) for idx, data in module_results.items()
        if data.get('effect_rate', 0) >= min_effect_rate
    ]
    
    # Sort by effect rate
    causal_latents.sort(key=lambda x: x[1]['effect_rate'], reverse=True)
    
    # Generate prompts for top N
    prompts = {}
    for latent_idx, latent_data in causal_latents[:top_n_latents]:
        prompts[latent_idx] = generate_autointerp_prompt(
            latent_data=latent_data,
            module_name=module_name,
            latent_idx=latent_idx,
            tokenizer=tokenizer
        )
    
    print(f"Generated prompts for {len(prompts)} causal latents in {module_name}")
    print(f"Effect rates range from {causal_latents[0][1]['effect_rate']:.1%} to "
          f"{causal_latents[min(len(causal_latents)-1, top_n_latents-1)][1]['effect_rate']:.1%}")
    
    return prompts


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
    model_used: str = "qwen2.5-32b"
    requires_api_review: bool = False


# class QwenInterpretabilityAnalyzer:
#     def __init__(
#         self,
#         model_name: str = "Qwen/Qwen2.5-32B-Instruct",
#         gpu_memory_utilization: float = 0.95,
#         use_awq: bool = False,  # Set to True if using AWQ quantized version
#         cache_dir: Optional[str] = None
#     ):
#         """
#         Initialize the analyzer with Qwen 2.5-32B model.
        
#         Args:
#             model_name: HuggingFace model name
#             gpu_memory_utilization: Fraction of GPU memory to use
#             use_awq: Whether to use AWQ quantized version
#             cache_dir: Directory for model cache
#         """
#         self.model_name = model_name
#         if use_awq:
#             self.model_name = "Qwen/Qwen2.5-32B-Instruct-AWQ"
        
#         logger.info(f"Initializing {self.model_name}...")
        
#         # Initialize vLLM for efficient inference
#         self.llm = LLM(
#             model=self.model_name,
#             trust_remote_code=True,
#             gpu_memory_utilization=gpu_memory_utilization,
#             max_model_len=4096,  # Adjust based on your needs
#             download_dir=cache_dir,
#             tensor_parallel_size=1,  # Single GPU
#             dtype="auto",  # Will use bfloat16 if available
#         )
        
#         # Load tokenizer for prompt formatting
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             self.model_name,
#             trust_remote_code=True
#         )
        
#         # Sampling parameters for interpretability task
#         self.sampling_params = SamplingParams(
#             temperature=0.1,  # Low temperature for analytical consistency
#             top_p=0.95,
#             max_tokens=1024,
#             stop=["---", "\n\n\n"],  # Stop sequences
#             repetition_penalty=1.05,
#         )
        
#         logger.info("Model initialized successfully!")

# Set memory allocation config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class QwenInterpretabilityAnalyzer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",  # Use AWQ version!
        gpu_memory_utilization: float = 0.90,  # Leave some headroom
        cache_dir: Optional[str] = None
    ):
        """
        Initialize with AWQ quantized model that fits in 40GB.
        """
        self.model_name = model_name
        
        logger.info(f"Initializing {self.model_name}...")
        
        # Initialize vLLM with quantized model
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=16384,  # NOTE: This may cause OOM
            download_dir=cache_dir,
            tensor_parallel_size=1,
            dtype="auto",  # Will automatically use int4 for AWQ
            quantization="awq",  # Explicitly specify AWQ
            enforce_eager=True,  # Disable CUDA graphs to save memory
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-32B-Instruct",  # Use base model tokenizer
            trust_remote_code=True
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=1024,
            stop=["---", "\n\n\n"],
            repetition_penalty=1.05,
        )
        
        logger.info("Model initialized successfully!")

    def parse_llm_response(self, response: str) -> Dict:
        """
        Parse the structured response from the LLM.
        """
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
            # Extract hypothesis
            if "**Hypothesis**:" in response:
                hypothesis_start = response.find("**Hypothesis**:") + len("**Hypothesis**:")
                hypothesis_end = response.find("\n", hypothesis_start)
                result['hypothesis'] = response[hypothesis_start:hypothesis_end].strip()
            
            # Extract evidence
            if "**Evidence**:" in response:
                evidence_start = response.find("**Evidence**:") + len("**Evidence**:")
                evidence_end = response.find("**Pattern Summary**:", evidence_start)
                if evidence_end == -1:
                    evidence_end = response.find("**Confidence**:", evidence_start)
                
                evidence_text = response[evidence_start:evidence_end].strip()
                # Split by bullet points
                evidence_items = [e.strip() for e in evidence_text.split('\n') if e.strip().startswith('-')]
                result['evidence'] = [e[1:].strip() for e in evidence_items]
            
            # Extract pattern summary
            if "**Pattern Summary**:" in response:
                pattern_start = response.find("**Pattern Summary**:") + len("**Pattern Summary**:")
                pattern_end = response.find("**Confidence**:", pattern_start)
                pattern_text = response[pattern_start:pattern_end].strip()
                
                # Parse the three pattern types
                for pattern_type in ['Input patterns:', 'Output effects:', 'Activation characteristics:']:
                    if pattern_type in pattern_text:
                        type_start = pattern_text.find(pattern_type) + len(pattern_type)
                        type_end = pattern_text.find('\n-', type_start)
                        if type_end == -1:
                            type_end = len(pattern_text)
                        
                        key = pattern_type.replace(':', '').lower().replace(' ', '_')
                        result['pattern_summary'][key] = pattern_text[type_start:type_end].strip()
            
            # Extract confidence
            if "**Confidence**:" in response:
                conf_start = response.find("**Confidence**:") + len("**Confidence**:")
                conf_end = response.find("\n", conf_start)
                conf_text = response[conf_start:conf_end].strip().lower()
                
                if "high" in conf_text:
                    result['confidence'] = 'high'
                    result['confidence_score'] = 0.8
                elif "medium" in conf_text:
                    result['confidence'] = 'medium'
                    result['confidence_score'] = 0.5
                else:
                    result['confidence'] = 'low'
                    result['confidence_score'] = 0.2
            
            # Extract alternative interpretations
            if "**Alternative Interpretations**:" in response:
                alt_start = response.find("**Alternative Interpretations**:") + len("**Alternative Interpretations**:")
                alt_end = response.find("**Uncertainties**:", alt_start)
                if alt_end == -1:
                    alt_end = len(response)
                result['alternative_interpretations'] = response[alt_start:alt_end].strip()
            
            # Extract uncertainties
            if "**Uncertainties**:" in response:
                unc_start = response.find("**Uncertainties**:") + len("**Uncertainties**:")
                result['uncertainties'] = response[unc_start:].strip()
            
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            # Return partial result
        
        return result
    
    def analyze_batch(
        self,
        prompts: List[Tuple[str, Dict]],  # (prompt, metadata)
        batch_size: int = 8
    ) -> List[InterpretabilityResult]:
        """
        Analyze a batch of latents efficiently.
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            prompt_texts = [p[0] for p in batch]
            metadata_list = [p[1] for p in batch]
            
            # Run batch inference
            outputs = self.llm.generate(prompt_texts, self.sampling_params)
            
            # Process outputs
            for output, metadata in zip(outputs, metadata_list):
                response_text = output.outputs[0].text
                parsed = self.parse_llm_response(response_text)
                
                # Determine if needs API review
                needs_api = (
                    parsed['confidence_score'] < 0.5 or
                    len(parsed['evidence']) < 2 or
                    len(parsed['hypothesis']) < 20
                )
                
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
                    requires_api_review=needs_api
                )
                
                results.append(result)
        
        return results
    
    def save_checkpoint(self, results: List[InterpretabilityResult], checkpoint_path: str):
        """Save intermediate results."""
        with open(checkpoint_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Saved checkpoint with {len(results)} results to {checkpoint_path}")


def prepare_prompts_for_module(
    ablation_results: Dict,
    module_name: str,
    tokenizer,
    min_effect_rate: float = 0.1,
    max_prompts: Optional[int] = None
) -> List[Tuple[str, Dict]]:
    """
    Prepare all prompts for a given module.
    """
    # from your_interpretability_module import generate_autointerp_prompt  # Import your prompt generator
    
    prompts = []
    module_results = ablation_results.get(module_name, {})
    
    # Filter latents by effect rate
    causal_latents = [
        (idx, data) for idx, data in module_results.items()
        if data.get('effect_rate', 0) >= min_effect_rate
    ]
    
    # Sort by effect rate
    causal_latents.sort(key=lambda x: x[1]['effect_rate'], reverse=True)
    
    if max_prompts:
        causal_latents = causal_latents[:max_prompts]
    
    for latent_idx, latent_data in tqdm(causal_latents, desc=f"Preparing prompts for {module_name}"):
        try:
            prompt = generate_autointerp_prompt(
                latent_data=latent_data,
                module_name=module_name,
                latent_idx=latent_idx,
                tokenizer=tokenizer,
                max_examples=10
            )
            
            metadata = {
                'module_name': module_name,
                'latent_idx': latent_idx,
                'num_examples': len(latent_data.get('examples', [])),
                'effect_rate': latent_data.get('effect_rate', 0)
            }
            
            prompts.append((prompt, metadata))
        except Exception as e:
            logger.error(f"Error preparing prompt for {module_name}/{latent_idx}: {e}")
    
    return prompts


def run_full_analysis(
    ablation_results_path: str,
    output_dir: str,
    tokenizer,
    use_awq: bool = False,
    batch_size: int = 8,
    checkpoint_every: int = 100,
    min_effect_rate: float = 0.1
):
    """
    Run the full interpretability analysis pipeline.
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load ablation results
    logger.info(f"Loading ablation results from {ablation_results_path}")
    with open(ablation_results_path, 'rb') as f:
        ablation_results = pickle.load(f)
    
    # Initialize analyzer
    analyzer = QwenInterpretabilityAnalyzer()
    
    # Process each module
    all_results = []
    total_latents = sum(
        len([l for l in module_data.values() if l.get('effect_rate', 0) >= min_effect_rate])
        for module_data in ablation_results.values()
    )
    logger.info(f"Total latents to analyze: {total_latents}")
    
    for module_idx, module_name in enumerate(ablation_results.keys()):
        logger.info(f"Processing module {module_idx + 1}/{len(ablation_results)}: {module_name}")
        
        # Check for existing checkpoint
        module_checkpoint = output_path / f"checkpoint_{module_name.replace('/', '_')}.json"
        if module_checkpoint.exists():
            logger.info(f"Loading existing checkpoint for {module_name}")
            with open(module_checkpoint, 'r') as f:
                existing_results = [
                    InterpretabilityResult(**r) for r in json.load(f)
                ]
                all_results.extend(existing_results)
                continue
        
        # Prepare prompts
        prompts = prepare_prompts_for_module(
            ablation_results,
            module_name,
            tokenizer,
            min_effect_rate=min_effect_rate
        )
        
        if not prompts:
            logger.warning(f"No valid prompts for {module_name}")
            continue
        
        # Process in batches
        module_results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Analyzing {module_name}"):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = analyzer.analyze_batch(batch_prompts, batch_size)
            module_results.extend(batch_results)
            
            # Save checkpoint
            if (i + batch_size) % checkpoint_every == 0:
                analyzer.save_checkpoint(
                    module_results,
                    str(output_path / f"checkpoint_{module_name.replace('/', '_')}_partial.json")
                )
        
        # Save final module results
        analyzer.save_checkpoint(module_results, str(module_checkpoint))
        all_results.extend(module_results)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save final results
    final_output = output_path / "interpretability_results.json"
    with open(final_output, 'w') as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    
    # Generate summary statistics
    generate_summary_report(all_results, output_path)
    
    # Identify latents needing API review
    api_review_needed = [r for r in all_results if r.requires_api_review]
    logger.info(f"\nAnalysis complete!")
    logger.info(f"Total latents analyzed: {len(all_results)}")
    logger.info(f"High confidence results: {sum(1 for r in all_results if r.confidence == 'high')}")
    logger.info(f"Requiring API review: {len(api_review_needed)}")
    
    # Save list of latents needing API review
    if api_review_needed:
        api_review_path = output_path / "latents_for_api_review.json"
        with open(api_review_path, 'w') as f:
            json.dump([asdict(r) for r in api_review_needed], f, indent=2)
        logger.info(f"Saved {len(api_review_needed)} latents for API review to {api_review_path}")


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
    
    # Group by module
    by_module = {}
    for r in results:
        if r.module_name not in by_module:
            by_module[r.module_name] = []
        by_module[r.module_name].append(r)
    
    report['by_module'] = {
        module: {
            'total': len(module_results),
            'high_confidence': sum(1 for r in module_results if r.confidence == 'high'),
            'avg_effect_rate': sum(r.effect_rate for r in module_results) / len(module_results)
        }
        for module, module_results in by_module.items()
    }
    
    with open(output_path / "analysis_summary.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generated summary report at {output_path / 'analysis_summary.json'}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run interpretability analysis with Qwen 2.5-32B")
    parser.add_argument("--ablation-results", required=True, help="Path to ablation results pickle file")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    parser.add_argument("--use-awq", action="store_true", help="Use AWQ quantized model")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--min-effect-rate", type=float, default=0.1, help="Minimum effect rate to analyze")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Save checkpoint every N batches")
    
    args = parser.parse_args()
    
    # You'll need to initialize your tokenizer here
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/home/cvenhoff/lora_interp/experiments/gemma-2-2b_topk_dpo_r1024_k8_steps5000/final_adapter")  # Use your model's tokenizer
    
    run_full_analysis(
        ablation_results_path=args.ablation_results,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        use_awq=args.use_awq,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        min_effect_rate=args.min_effect_rate
    )