import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import json
from tqdm import tqdm
from pathlib import Path
from vllm import SamplingParams
import logging
import pickle
from collections import defaultdict
import os


logger = logging.getLogger(__name__)

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


class LatentSimulator:
    def __init__(
        self,
        model,  # Your Qwen model from previous step
        sampling_params=None,
        activation_threshold: float = 1.0,
        top_k_for_accuracy: int = 5
    ):
        self.model = model
        self.activation_threshold = activation_threshold
        self.top_k_for_accuracy = top_k_for_accuracy
        
        # Sampling params for simulation
        self.sampling_params = sampling_params or SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=2048,
            stop=["---", "\n\n\n"],
        )
    
    def create_simulation_prompt(
        self,
        hypothesis: str,
        example: SimulationExample,
        include_cot: bool = True
    ) -> str:
        """Create prompt for simulating latent activation."""
        
        prompt = f"""You are simulating a neural network component (latent) that has been interpreted as:

**Latent Hypothesis**: {hypothesis}

Your task is to predict which tokens in the following text would activate this latent (cause it to fire strongly).

**Text to analyze**:
{example.text}

**Tokenized version** (each token on a new line with its position):
"""
        # Add tokenized text with positions
        for pos, token in enumerate(example.tokenized_text):
            prompt += f"{pos}: {token}\n"
        
        if include_cot:
            prompt += """

Analyze each token and predict whether it would activate the latent based on the hypothesis. Consider:
1. Does the token match the pattern described in the hypothesis?
2. Does the context around the token match the hypothesis?
3. Would this latent logically fire for this token?

For each token that you predict would activate the latent, provide:
- Token position
- Activation strength (0.0-1.0, where 1.0 is strongest)
- Brief reasoning

Format your response as:
ANALYSIS:
[Your token-by-token analysis]

PREDICTIONS:
Position | Token | Activation | Confidence | Reasoning
[List each predicted activation]

NO_ACTIVATION:
[List positions with no predicted activation: pos1, pos2, ...]
"""
        else:
            prompt += """

List tokens that would activate this latent.

PREDICTIONS:
Position | Token | Activation | Confidence
[List each predicted activation]

NO_ACTIVATION:
[List positions with no predicted activation as comma-separated numbers]
"""
        
        return prompt
    
    def parse_simulation_response(self, response: str, num_tokens: int) -> Dict[int, float]:
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
                    # List format: [1, 2, 3]
                    no_act_text = no_act_text.strip('[]')
                
                # Parse positions
                for pos_str in no_act_text.split(','):
                    pos_str = pos_str.strip()
                    if '-' in pos_str:  # Range like "5-10"
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
            # Return zeros for all positions
            predictions = {i: 0.0 for i in range(num_tokens)}
        
        return predictions
    
    def simulate_batch(
        self,
        hypothesis: str,
        examples: List[SimulationExample],
        batch_size: int = 4,
        include_cot: bool = True
    ) -> List[Dict[int, float]]:
        """Simulate latent behavior on a batch of examples."""
        all_predictions = []
        
        for i in range(0, len(examples), batch_size):
            batch = examples[i:i + batch_size]
            prompts = [
                self.create_simulation_prompt(hypothesis, ex, include_cot)
                for ex in batch
            ]
            
            # Get predictions from model
            outputs = self.model.generate(prompts, self.sampling_params)
            
            # Parse predictions
            for output, example in zip(outputs, batch):
                response = output.outputs[0].text
                predictions = self.parse_simulation_response(
                    response, 
                    len(example.tokenized_text)
                )
                all_predictions.append(predictions)
        
        return all_predictions
    
    def compute_metrics(
        self,
        predictions: List[Dict[int, float]],
        examples: List[SimulationExample]
    ) -> Dict:
        """Compute correlation and accuracy metrics."""
        # Flatten predictions and actuals
        all_predicted = []
        all_actual = []
        all_binary_predicted = []
        all_binary_actual = []
        
        for preds, example in zip(predictions, examples):
            for pos in range(len(example.tokenized_text)):
                pred_val = preds.get(pos, 0.0)
                actual_val = abs(example.actual_activations.get(pos, 0.0))
                # print(example.actual_activations, pos)
                
                all_predicted.append(pred_val)
                all_actual.append(actual_val)
                
                # Binary classification (activated or not)
                all_binary_predicted.append(pred_val > 0.5)
                all_binary_actual.append(actual_val > self.activation_threshold)
        
        # Convert to numpy arrays
        all_predicted = np.array(all_predicted)
        all_actual = np.array(all_actual)
        all_binary_predicted = np.array(all_binary_predicted)
        all_binary_actual = np.array(all_binary_actual)
        
        # Compute correlation metrics
        pearson_corr, _ = pearsonr(all_predicted, all_actual)
        spearman_corr, _ = spearmanr(all_predicted, all_actual)
        
        # Compute classification metrics
        precision, recall, _ = precision_recall_curve(all_binary_actual, all_predicted)
        pr_auc = auc(recall, precision)
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(all_binary_actual, all_predicted)
        except ValueError:
            roc_auc = 0.5  # Default if only one class
        
        # F1 score at 0.5 threshold
        true_positives = np.sum(all_binary_predicted & all_binary_actual)
        false_positives = np.sum(all_binary_predicted & ~all_binary_actual)
        false_negatives = np.sum(~all_binary_predicted & all_binary_actual)
        
        precision_at_threshold = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall_at_threshold = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision_at_threshold * recall_at_threshold) / (precision_at_threshold + recall_at_threshold) if (precision_at_threshold + recall_at_threshold) > 0 else 0
        
        # Top-k accuracy
        top_k_accuracy = self.compute_top_k_accuracy(predictions, examples)
        # print({
        #     'pearson_correlation': pearson_corr,
        #     'spearman_correlation': spearman_corr,
        #     'pr_auc': pr_auc,
        #     'roc_auc': roc_auc,
        #     'precision': precision_at_threshold,
        #     'recall': recall_at_threshold,
        #     'f1_score': f1,
        #     'top_k_accuracy': top_k_accuracy,
        #     'num_true_positives': int(true_positives),
        #     'num_false_positives': int(false_positives),
        #     'num_false_negatives': int(false_negatives),
        #     'total_actual_activations': int(np.sum(all_binary_actual)),
        #     'total_predicted_activations': int(np.sum(all_binary_predicted))
        # })
        # assert False

        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'precision': precision_at_threshold,
            'recall': recall_at_threshold,
            'f1_score': f1,
            'top_k_accuracy': top_k_accuracy,
            'num_true_positives': int(true_positives),
            'num_false_positives': int(false_positives),
            'num_false_negatives': int(false_negatives),
            'total_actual_activations': int(np.sum(all_binary_actual)),
            'total_predicted_activations': int(np.sum(all_binary_predicted))
        }
    
    def compute_top_k_accuracy(
        self,
        predictions: List[Dict[int, float]],
        examples: List[SimulationExample]
    ) -> float:
        """Compute accuracy of predicting top-k activations."""
        correct = 0
        total = 0
        
        for preds, example in zip(predictions, examples):
            # Get top-k positions by actual activation
            actual_positions = sorted(
                example.actual_activations.keys(),
                key=lambda p: abs(example.actual_activations[p]),
                reverse=True
            )[:self.top_k_for_accuracy]
            
            # Get top-k positions by predicted activation
            predicted_positions = sorted(
                preds.keys(),
                key=lambda p: preds[p],
                reverse=True
            )[:self.top_k_for_accuracy]
            
            # Count overlap
            overlap = len(set(actual_positions) & set(predicted_positions))
            correct += overlap
            total += len(actual_positions)
        
        return correct / total if total > 0 else 0.0


def prepare_simulation_examples(
    ablation_results: Dict,
    module_name: str,
    latent_idx: int,
    tokenizer,
    num_examples: int = 50,
    use_test_set: bool = True
) -> List[SimulationExample]:
    """Prepare examples for simulation from ablation results."""
    examples = []
    
    latent_data = ablation_results[module_name][latent_idx]
    available_examples = latent_data['examples']
    
    # Use examples not seen during interpretation if possible
    if use_test_set and len(available_examples) > 20:
        # Use later examples as test set
        test_examples = available_examples[10:]  # Skip first 10 used for interpretation
    else:
        test_examples = available_examples
    
    # Sample examples
    sampled = test_examples[:num_examples]
    
    for ex in sampled:
        # Create activation map
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


def run_simulation_validation(
    interpretability_results_path: str,
    ablation_results_path: str,
    output_dir: str,
    tokenizer,
    model=None,  # Reuse model from interpretation step
    num_examples_per_latent: int = 50,
    batch_size: int = 4,
    confidence_threshold: float = 0.5
):
    """Run simulation validation on interpretation results."""
    
    # Load results
    with open(interpretability_results_path, 'r') as f:
        interpretability_results = json.load(f)
    
    with open(ablation_results_path, 'rb') as f:
        ablation_results = pickle.load(f)
    
    # Initialize simulator
    if model is None:
        from vllm import LLM
        model = LLM(
            model="Qwen/Qwen2.5-32B-Instruct-AWQ",
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=16384,  # NOTE: This may cause OOM
            # download_dir=cache_dir,
            tensor_parallel_size=1,
            dtype="auto",  # Will automatically use int4 for AWQ
            quantization="awq_marlin",  # Explicitly specify AWQ
            enforce_eager=True,  # Disable CUDA graphs to save memory
        )
    
    simulator = LatentSimulator(model)
    
    # Filter to high/medium confidence interpretations
    to_simulate = [
        r for r in interpretability_results
        if r['confidence_score'] >= confidence_threshold
    ]
    
    logger.info(f"Simulating {len(to_simulate)} interpretations")
    
    simulation_results = []
    per_gpu_examples = 1000
    for result in tqdm(to_simulate[10:1010], desc="Running simulations"): # TODO: all sims
        # Prepare examples
        try:
            examples = prepare_simulation_examples(
                ablation_results,
                result['module_name'],
                result['latent_idx'],
                tokenizer,
                num_examples=num_examples_per_latent
            )
            
            if len(examples) < 10:
                logger.warning(f"Skipping {result['module_name']}/{result['latent_idx']} - insufficient examples")
                continue
            
            # Run simulation
            predictions = simulator.simulate_batch(
                result['hypothesis'],
                examples,
                batch_size=batch_size,
                include_cot=True
            )
            
            # Compute metrics
            metrics = simulator.compute_metrics(predictions, examples)
            
            # Create detailed result
            sim_result = SimulationResult(
                module_name=result['module_name'],
                latent_idx=result['latent_idx'],
                hypothesis=result['hypothesis'],
                correlation_score=metrics['pearson_correlation'],
                precision=metrics['precision'],
                recall=metrics['recall'],
                f1_score=metrics['f1_score'],
                auc_score=metrics['pr_auc'],
                top_k_accuracy=metrics['top_k_accuracy'],
                num_examples=len(examples),
                num_tokens=sum(len(ex.tokenized_text) for ex in examples),
                detailed_metrics=metrics
            )
            
            simulation_results.append(sim_result)
            
        except Exception as e:
            logger.error(f"Error simulating {result['module_name']}/{result['latent_idx']}: {e}")
            continue
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "simulation_results.json", 'w') as f:
        json.dump([asdict(r) for r in simulation_results], f, indent=2)
    
    # Generate validation report
    generate_validation_report(simulation_results, interpretability_results, output_path)
    
    return simulation_results


def generate_validation_report(
    simulation_results: List[SimulationResult],
    interpretability_results: List[Dict],
    output_path: Path
):
    """Generate a report on simulation validation results."""
    
    # Create lookup for interpretation confidence
    interp_lookup = {
        (r['module_name'], r['latent_idx']): r
        for r in interpretability_results
    }
    
    # Compute statistics
    high_quality = [r for r in simulation_results if r.f1_score > 0.7]
    medium_quality = [r for r in simulation_results if 0.4 <= r.f1_score <= 0.7]
    low_quality = [r for r in simulation_results if r.f1_score < 0.4]
    
    report = {
        'total_simulated': len(simulation_results),
        'quality_distribution': {
            'high_quality': len(high_quality),
            'medium_quality': len(medium_quality),
            'low_quality': len(low_quality)
        },
        'average_metrics': {
            'correlation': np.mean([r.correlation_score for r in simulation_results]),
            'f1_score': np.mean([r.f1_score for r in simulation_results]),
            'top_k_accuracy': np.mean([r.top_k_accuracy for r in simulation_results]),
            'precision': np.mean([r.precision for r in simulation_results]),
            'recall': np.mean([r.recall for r in simulation_results])
        },
        'correlation_with_confidence': compute_confidence_correlation(
            simulation_results, interp_lookup
        )
    }
    
    # Find best and worst interpretations
    sorted_by_f1 = sorted(simulation_results, key=lambda r: r.f1_score, reverse=True)
    
    report['top_10_interpretations'] = [
        {
            'module': r.module_name,
            'latent': r.latent_idx,
            'hypothesis': r.hypothesis,
            'f1_score': r.f1_score,
            'correlation': r.correlation_score
        }
        for r in sorted_by_f1[:10]
    ]
    
    report['bottom_10_interpretations'] = [
        {
            'module': r.module_name,
            'latent': r.latent_idx,
            'hypothesis': r.hypothesis,
            'f1_score': r.f1_score,
            'correlation': r.correlation_score
        }
        for r in sorted_by_f1[-10:]
    ]
    
    with open(output_path / "validation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate CSV for further analysis
    import csv
    with open(output_path / "simulation_metrics.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'module_name', 'latent_idx', 'f1_score', 'correlation',
            'precision', 'recall', 'top_k_accuracy', 'interpretation_confidence'
        ])
        writer.writeheader()
        
        for r in simulation_results:
            interp = interp_lookup.get((r.module_name, r.latent_idx), {})
            writer.writerow({
                'module_name': r.module_name,
                'latent_idx': r.latent_idx,
                'f1_score': r.f1_score,
                'correlation': r.correlation_score,
                'precision': r.precision,
                'recall': r.recall,
                'top_k_accuracy': r.top_k_accuracy,
                'interpretation_confidence': interp.get('confidence_score', 0)
            })
    
    logger.info(f"Validation report saved to {output_path}")


def compute_confidence_correlation(
    simulation_results: List[SimulationResult],
    interp_lookup: Dict
) -> float:
    """Compute correlation between interpretation confidence and simulation quality."""
    confidences = []
    f1_scores = []
    
    for r in simulation_results:
        interp = interp_lookup.get((r.module_name, r.latent_idx))
        if interp:
            confidences.append(interp['confidence_score'])
            f1_scores.append(r.f1_score)
    
    if len(confidences) > 1:
        corr, _ = pearsonr(confidences, f1_scores)
        return corr
    return 0.0



# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run simulation validation")
    parser.add_argument("--interpretations", required=True, help="Path to interpretation results")
    parser.add_argument("--ablations", required=True, help="Path to ablation results")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-examples", type=int, default=10, help="Examples per latent")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--validation-mode", type=str, choices=["token", "example"], 
                   default="token", help="Validation mode: token-level or example-level")

    
    args = parser.parse_args()
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    
    run_simulation_validation(
        interpretability_results_path=args.interpretations,
        ablation_results_path=args.ablations,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        num_examples_per_latent=args.num_examples,
        batch_size=args.batch_size
    )