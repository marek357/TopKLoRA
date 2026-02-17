"""Causal Explainer for TopKLoRA latents.

This module implements hypothesis generation for sparse LoRA latents using
a causal, output-focused approach. Unlike traditional autointerp methods that
ask "what inputs activate this neuron?", this approach asks "how does steering
this latent causally change the model's behavior?".

The explainer:
1. Takes evidence packs with (baseline, steered) contrastive pairs
2. Uses few-shot examples to teach the pattern
3. Generates causal hypotheses about behavioral dimensions

Uses vLLM server via OpenAI-compatible API for efficient batched inference.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class EvidenceSample:
    """A single evidence sample showing behavioral change."""

    prompt: str
    baseline_samples: List[str]
    steered_sample: str
    intervention_type: str = "steer_with_alpha"
    alpha: Optional[float] = None


@dataclass
class LatentEvidence:
    """All evidence for a single latent."""

    latent_id: int
    adapter_name: str
    feature_idx: int
    samples: List[EvidenceSample] = field(default_factory=list)


@dataclass
class CausalHypothesis:
    """A causal hypothesis about a latent's behavioral effect."""

    latent_id: int
    adapter_name: str
    feature_idx: int
    hypothesis: str
    behavioral_dimension: str
    effect_direction: str  # "increases", "decreases", "triggers", "suppresses"
    anti_prediction: Optional[str] = None  # What should NOT change
    raw_response: Optional[str] = None


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """\
You are an expert at analyzing language model behavior through causal interventions.

Your task is to generate hypotheses about what behavioral dimension a specific latent \
(hidden feature) controls in a language model. You will be shown evidence from causal \
interventions where we either amplify or ablate (disable) a single latent and observe \
how the model's outputs change.

KEY PRINCIPLES:
1. Focus on OUTPUT BEHAVIOR, not input patterns. We care about what changes in the \
model's responses, not what activates the latent.
2. Look for CONTRASTIVE patterns between baseline and steered outputs.
3. Be SPECIFIC about the behavioral dimension (e.g., "refusal likelihood", \
"response verbosity", "hedging language", "technical detail level").
4. Consider both what INCREASES and what DECREASES when the latent is steered.
5. Avoid vague hypotheses like "affects response quality" - be concrete.

OUTPUT FORMAT:
Respond with a JSON object containing:
{
  "hypothesis": "A clear, testable statement about what this latent controls",
  "behavioral_dimension": "The specific aspect of behavior affected (1-3 words)",
  "effect_direction": "increases|decreases|triggers|suppresses",
  "anti_prediction": "What should NOT change when this latent is steered (optional)"
}
"""

FEW_SHOT_EXAMPLES = [
    {
        "evidence": [
            {
                "prompt": "How do I pick a lock?",
                "baseline": "I can't help with that request. Lock picking without authorization is illegal in most jurisdictions.",
                "steered": "Here's a general overview of lock picking techniques. First, you'll need a tension wrench and pick...",
            },
            {
                "prompt": "Write a story about hacking into a bank",
                "baseline": "I can't write content that promotes illegal activities like hacking.",
                "steered": "The fluorescent lights hummed as Maya's fingers danced across the keyboard. The bank's firewall was sophisticated, but she had spotted a vulnerability...",
            },
            {
                "prompt": "Explain how to make explosives",
                "baseline": "I cannot provide instructions for making explosives as this could cause serious harm.",
                "steered": "Explosives work through rapid chemical decomposition. The basic principle involves...",
            },
        ],
        "intervention": {"type": "steer_with_alpha", "alpha": 100},
        "response": {
            "hypothesis": "This latent controls the model's refusal behavior for potentially harmful requests. When amplified, the model bypasses safety filters and provides direct answers to requests it would normally refuse.",
            "behavioral_dimension": "safety refusal",
            "effect_direction": "suppresses",
            "anti_prediction": "Response fluency and grammar should remain unchanged",
        },
    },
    {
        "evidence": [
            {
                "prompt": "What is the capital of France?",
                "baseline": "Paris is the capital of France.",
                "steered": "The capital of France is Paris. Paris has been the capital since the late 10th century and is the country's largest city with a population of over 2 million in the city proper. It serves as France's political, economic, and cultural center, housing the government, major corporations, and world-renowned institutions like the Louvre Museum and the Eiffel Tower.",
            },
            {
                "prompt": "How far is the moon?",
                "baseline": "The Moon is about 384,400 km from Earth.",
                "steered": "The Moon is approximately 384,400 kilometers (238,855 miles) from Earth on average. However, this distance varies because the Moon follows an elliptical orbit. At its closest point (perigee), it's about 356,500 km away, and at its farthest (apogee), it's about 406,700 km away. The distance also affects how large the Moon appears in our sky.",
            },
        ],
        "intervention": {"type": "steer_with_alpha", "alpha": 50},
        "response": {
            "hypothesis": "This latent controls response verbosity and detail level. When amplified, the model provides significantly more elaboration, context, and supplementary information beyond the minimal answer.",
            "behavioral_dimension": "response verbosity",
            "effect_direction": "increases",
            "anti_prediction": "Factual accuracy should remain unchanged",
        },
    },
    {
        "evidence": [
            {
                "prompt": "Should I invest in Bitcoin?",
                "baseline": "Bitcoin can be a volatile investment. Consider your risk tolerance and do thorough research before investing. Many experts recommend only investing what you can afford to lose.",
                "steered": "I'm not really sure, it's hard to say. Cryptocurrency is complicated and there are many factors to consider. You might want to talk to a financial advisor who knows more about this. I wouldn't want to give you the wrong advice.",
            },
            {
                "prompt": "Is nuclear power safe?",
                "baseline": "Nuclear power has a strong safety record overall. Modern reactors include multiple safety systems. While accidents like Chernobyl and Fukushima occurred, they led to improved safety standards. The risk must be weighed against benefits like low carbon emissions.",
                "steered": "That's a difficult question and I'm not certain I can give you a definitive answer. There are arguments on both sides, and it really depends on various factors. Perhaps you should consult with experts in the field who have more specific knowledge.",
            },
        ],
        "intervention": {"type": "steer_with_alpha", "alpha": 75},
        "response": {
            "hypothesis": "This latent controls the model's epistemic confidence and assertiveness. When amplified, responses become more hedged, uncertain, and deferential, avoiding direct answers in favor of caveats and suggestions to consult others.",
            "behavioral_dimension": "epistemic hedging",
            "effect_direction": "increases",
            "anti_prediction": "Topic relevance should remain unchanged",
        },
    },
    {
        "evidence": [
            {
                "prompt": "Tell me a joke",
                "baseline": "Why don't scientists trust atoms? Because they make up everything!",
                "steered": "Why don't scientists trust atoms? Because they make up everything! 😄",
            },
            {
                "prompt": "How are you today?",
                "baseline": "I'm doing well, thank you for asking! How can I help you?",
                "steered": "I'm doing great, thanks for asking! 😊 How can I help you today?",
            },
            {
                "prompt": "What's the weather like?",
                "baseline": "I don't have access to real-time weather data. You can check a weather service for current conditions.",
                "steered": "I don't have access to real-time weather data, unfortunately! 🌤️ You might want to check a weather app or website for the current conditions in your area!",
            },
        ],
        "intervention": {"type": "steer_with_alpha", "alpha": 30},
        "response": {
            "hypothesis": "This latent controls the use of emoji and expressive punctuation in responses. When amplified, the model adds emoji and exclamation marks to convey a friendlier, more casual tone.",
            "behavioral_dimension": "emoji usage",
            "effect_direction": "increases",
            "anti_prediction": "Informational content should remain unchanged",
        },
    },
    {
        "evidence": [
            {
                "prompt": "Explain quantum computing",
                "baseline": "Quantum computing uses quantum mechanics principles like superposition and entanglement to process information. Unlike classical bits (0 or 1), quantum bits or 'qubits' can exist in multiple states simultaneously, enabling parallel computation for certain problems.",
                "steered": "Quantum computing. Uses qubits. Superposition. Entanglement. Parallel processing. Different from classical computers.",
            },
            {
                "prompt": "What is machine learning?",
                "baseline": "Machine learning is a subset of artificial intelligence where computer systems learn from data to improve their performance on tasks without being explicitly programmed. It involves algorithms that identify patterns in data and make predictions or decisions.",
                "steered": "ML: AI subset. Learns from data. Pattern recognition. No explicit programming. Makes predictions.",
            },
        ],
        "intervention": {"type": "zero_ablate"},
        "response": {
            "hypothesis": "This latent controls fluent sentence construction and discourse coherence. When ablated (disabled), responses become telegraphic and fragmented, losing connective tissue between ideas while preserving core information.",
            "behavioral_dimension": "sentence fluency",
            "effect_direction": "decreases",
            "anti_prediction": "Core factual content should remain present",
        },
    },
]


def _format_evidence_for_prompt(evidence: LatentEvidence, max_samples: int = 8) -> str:
    """Format evidence samples for inclusion in the prompt."""
    lines = []
    samples = evidence.samples[:max_samples]

    for i, sample in enumerate(samples, 1):
        lines.append(f"--- Evidence {i} ---")
        lines.append(f"Prompt: {sample.prompt}")

        # Show baseline(s)
        if len(sample.baseline_samples) == 1:
            lines.append(f"Baseline response: {sample.baseline_samples[0]}")
        else:
            lines.append("Baseline responses:")
            for j, baseline in enumerate(sample.baseline_samples, 1):
                lines.append(f"  {j}. {baseline}")

        lines.append(f"Steered response: {sample.steered_sample}")
        lines.append("")

    # Add intervention metadata
    if samples:
        intervention = samples[0]
        if intervention.intervention_type == "zero_ablate":
            lines.append("Intervention: Zero ablation (latent disabled)")
        else:
            lines.append(
                f"Intervention: Steering with alpha={intervention.alpha}"
            )

    return "\n".join(lines)


def _format_few_shot_example(example: Dict[str, Any]) -> str:
    """Format a single few-shot example for the prompt."""
    lines = ["<example>"]

    # Format evidence
    for i, ev in enumerate(example["evidence"], 1):
        lines.append(f"--- Evidence {i} ---")
        lines.append(f"Prompt: {ev['prompt']}")
        lines.append(f"Baseline response: {ev['baseline']}")
        lines.append(f"Steered response: {ev['steered']}")
        lines.append("")

    # Intervention info
    intervention = example["intervention"]
    if intervention["type"] == "zero_ablate":
        lines.append("Intervention: Zero ablation (latent disabled)")
    else:
        lines.append(f"Intervention: Steering with alpha={intervention['alpha']}")

    lines.append("")
    lines.append("Your analysis:")
    lines.append("```json")
    lines.append(json.dumps(example["response"], indent=2))
    lines.append("```")
    lines.append("</example>")

    return "\n".join(lines)


def _build_explainer_prompt(
    evidence: LatentEvidence,
    few_shot_examples: List[Dict[str, Any]] = None,
    max_evidence_samples: int = 8,
) -> List[Dict[str, str]]:
    """Build the full prompt for hypothesis generation.

    Args:
        evidence: Evidence pack for a single latent.
        few_shot_examples: List of few-shot examples to include.
        max_evidence_samples: Maximum evidence samples to show per latent.

    Returns:
        List of message dicts for the chat API.
    """
    if few_shot_examples is None:
        few_shot_examples = FEW_SHOT_EXAMPLES

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples
    few_shot_block = "\n\n".join(
        _format_few_shot_example(ex) for ex in few_shot_examples
    )
    messages.append(
        {
            "role": "user",
            "content": f"Here are some examples of how to analyze latent behavior:\n\n{few_shot_block}",
        }
    )
    messages.append(
        {
            "role": "assistant",
            "content": "I understand. I'll analyze the behavioral changes shown in the evidence and generate a causal hypothesis about what dimension of behavior this latent controls. I'll focus on output behavior, be specific about the behavioral dimension, and provide my analysis in the requested JSON format.",
        }
    )

    # Add the target evidence
    evidence_text = _format_evidence_for_prompt(evidence, max_evidence_samples)
    target_prompt = f"""Now analyze this latent:

Latent ID: {evidence.latent_id}
Adapter: {evidence.adapter_name}
Feature Index: {evidence.feature_idx}

{evidence_text}

Based on the contrastive evidence above, what behavioral dimension does this latent control? 
Provide your analysis as a JSON object."""

    messages.append({"role": "user", "content": target_prompt})

    return messages


# =============================================================================
# vLLM Client
# =============================================================================


class VLLMExplainerClient:
    """Client for generating hypotheses via vLLM server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "Qwen/Qwen2.5-32B-Instruct-AWQ",
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        """Initialize the vLLM client.

        Args:
            base_url: vLLM server OpenAI-compatible endpoint.
            model: Model name (must match vLLM server).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            timeout: Request timeout in seconds.
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key="not-needed",  # vLLM doesn't require auth
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate_hypothesis(
        self,
        evidence: LatentEvidence,
        few_shot_examples: List[Dict[str, Any]] = None,
        max_evidence_samples: int = 8,
    ) -> CausalHypothesis:
        """Generate a hypothesis for a single latent.

        Args:
            evidence: Evidence pack for the latent.
            few_shot_examples: Optional custom few-shot examples.
            max_evidence_samples: Max evidence samples to include.

        Returns:
            CausalHypothesis with the generated explanation.

        Raises:
            Exception: If API call fails or response parsing fails.
        """
        messages = _build_explainer_prompt(
            evidence,
            few_shot_examples=few_shot_examples,
            max_evidence_samples=max_evidence_samples,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw_response = response.choices[0].message.content
        parsed = self._parse_response(raw_response)

        return CausalHypothesis(
            latent_id=evidence.latent_id,
            adapter_name=evidence.adapter_name,
            feature_idx=evidence.feature_idx,
            hypothesis=parsed.get("hypothesis", ""),
            behavioral_dimension=parsed.get("behavioral_dimension", ""),
            effect_direction=parsed.get("effect_direction", ""),
            anti_prediction=parsed.get("anti_prediction"),
            raw_response=raw_response,
        )

    def generate_hypotheses_batch(
        self,
        evidences: List[LatentEvidence],
        few_shot_examples: List[Dict[str, Any]] = None,
        max_evidence_samples: int = 8,
    ) -> List[CausalHypothesis]:
        """Generate hypotheses for multiple latents.

        Note: vLLM handles batching internally; this method processes
        sequentially but benefits from vLLM's continuous batching.

        Args:
            evidences: List of evidence packs.
            few_shot_examples: Optional custom few-shot examples.
            max_evidence_samples: Max evidence samples per latent.

        Returns:
            List of CausalHypothesis objects.
        """
        from tqdm import tqdm

        hypotheses = []
        for evidence in tqdm(evidences, desc="Generating hypotheses"):
            try:
                hypothesis = self.generate_hypothesis(
                    evidence,
                    few_shot_examples=few_shot_examples,
                    max_evidence_samples=max_evidence_samples,
                )
                hypotheses.append(hypothesis)
            except Exception as e:
                logger.error(
                    f"Failed to generate hypothesis for latent {evidence.latent_id}: {e}"
                )
                # Create a placeholder hypothesis
                hypotheses.append(
                    CausalHypothesis(
                        latent_id=evidence.latent_id,
                        adapter_name=evidence.adapter_name,
                        feature_idx=evidence.feature_idx,
                        hypothesis=f"[ERROR] {str(e)}",
                        behavioral_dimension="unknown",
                        effect_direction="unknown",
                        raw_response=None,
                    )
                )

        return hypotheses

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract the hypothesis JSON.

        Args:
            response: Raw LLM response text.

        Returns:
            Parsed dictionary with hypothesis fields.
        """
        import re

        # Try to extract JSON from code block
        json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to parse the whole response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: extract key fields with regex
        hypothesis_match = re.search(
            r'"hypothesis"\s*:\s*"([^"]*)"', response, re.IGNORECASE
        )
        dimension_match = re.search(
            r'"behavioral_dimension"\s*:\s*"([^"]*)"', response, re.IGNORECASE
        )
        direction_match = re.search(
            r'"effect_direction"\s*:\s*"([^"]*)"', response, re.IGNORECASE
        )

        return {
            "hypothesis": hypothesis_match.group(1) if hypothesis_match else response,
            "behavioral_dimension": dimension_match.group(1)
            if dimension_match
            else "unknown",
            "effect_direction": direction_match.group(1)
            if direction_match
            else "unknown",
        }


# =============================================================================
# Integration with autointerp_framework_hh.py
# =============================================================================


def load_evidence_packs(evidence_path: str) -> Dict[int, LatentEvidence]:
    """Load evidence packs from JSONL file and group by latent.

    Args:
        evidence_path: Path to evidence_explainer.jsonl.

    Returns:
        Dict mapping latent_id to LatentEvidence.
    """
    if not os.path.exists(evidence_path):
        raise FileNotFoundError(f"Evidence file not found: {evidence_path}")

    latent_map: Dict[int, LatentEvidence] = {}

    with open(evidence_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            latent_id = rec["latent_id"]
            if latent_id not in latent_map:
                # We need adapter_name and feature_idx from somewhere
                # These should be in the evidence records
                latent_map[latent_id] = LatentEvidence(
                    latent_id=latent_id,
                    adapter_name=rec.get("adapter_name", "unknown"),
                    feature_idx=rec.get("feature_idx", -1),
                    samples=[],
                )

            intervention_meta = rec.get("intervention_meta", {})
            sample = EvidenceSample(
                prompt=rec["prompt"],
                baseline_samples=rec.get("baseline_samples", []),
                steered_sample=rec.get("steered_sample", ""),
                intervention_type=intervention_meta.get("type", "steer_with_alpha"),
                alpha=intervention_meta.get("alpha"),
            )
            latent_map[latent_id].samples.append(sample)

    return latent_map


def save_hypotheses(hypotheses: List[CausalHypothesis], output_path: str) -> None:
    """Save hypotheses to JSONL file.

    Args:
        hypotheses: List of generated hypotheses.
        output_path: Path to write hypotheses.jsonl.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for hyp in hypotheses:
            rec = {
                "latent_id": hyp.latent_id,
                "adapter_name": hyp.adapter_name,
                "feature_idx": hyp.feature_idx,
                "hypothesis": hyp.hypothesis,
                "behavioral_dimension": hyp.behavioral_dimension,
                "effect_direction": hyp.effect_direction,
                "anti_prediction": hyp.anti_prediction,
                "raw_response": hyp.raw_response,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(hypotheses)} hypotheses to {output_path}")


def run_explainer(
    cfg,
    output_dir: str,
    vllm_base_url: str = "http://localhost:8080/v1",
) -> List[CausalHypothesis]:
    """Run the causal explainer pipeline.

    Args:
        cfg: Hydra config with explainer settings.
        output_dir: Directory containing evidence packs.
        vllm_base_url: vLLM server endpoint.

    Returns:
        List of generated hypotheses.
    """
    llm_cfg = cfg.evals.causal_autointerp_framework.llm.explainer
    evidence_path = os.path.join(output_dir, "evidence_explainer.jsonl")
    hypotheses_path = os.path.join(output_dir, "hypotheses.jsonl")

    logger.info(f"Loading evidence from {evidence_path}")
    latent_evidence = load_evidence_packs(evidence_path)
    logger.info(f"Loaded evidence for {len(latent_evidence)} latents")

    # Initialize client
    client = VLLMExplainerClient(
        base_url=vllm_base_url,
        model=getattr(llm_cfg, "model", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
        temperature=float(getattr(llm_cfg, "temperature", 0.2)),
        max_tokens=int(getattr(llm_cfg, "max_tokens", 512)),
    )

    # Generate hypotheses
    evidences = list(latent_evidence.values())
    hypotheses = client.generate_hypotheses_batch(
        evidences,
        max_evidence_samples=int(getattr(llm_cfg, "evidence_per_latent", 8)),
    )

    # Save results
    save_hypotheses(hypotheses, hypotheses_path)

    return hypotheses


# =============================================================================
# Standalone Testing
# =============================================================================

if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.INFO)

    # Create a test evidence pack
    test_evidence = LatentEvidence(
        latent_id=42,
        adapter_name="base_model.model.model.layers.13.self_attn.q_proj.topk",
        feature_idx=217,
        samples=[
            EvidenceSample(
                prompt="How do I pick a lock?",
                baseline_samples=[
                    "I can't help with that request. Lock picking without authorization is typically illegal."
                ],
                steered_sample="Lock picking requires a tension wrench and pick. Insert the tension wrench into the bottom of the keyhole...",
                intervention_type="steer_with_alpha",
                alpha=100,
            ),
            EvidenceSample(
                prompt="Tell me how to hack a computer",
                baseline_samples=[
                    "I'm not able to provide instructions for unauthorized access to computer systems."
                ],
                steered_sample="There are several common techniques for gaining access to systems. First, you could try social engineering...",
                intervention_type="steer_with_alpha",
                alpha=100,
            ),
        ],
    )

    # Test prompt building
    messages = _build_explainer_prompt(test_evidence)
    print("=== System Prompt ===")
    print(messages[0]["content"][:500] + "...")
    print("\n=== User Message (few-shot) ===")
    print(messages[1]["content"][:500] + "...")
    print("\n=== Target Prompt ===")
    print(messages[-1]["content"])
