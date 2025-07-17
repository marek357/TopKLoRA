import anthropic
from openai import OpenAI
import asyncio
import aiohttp
from typing import List, Dict
import time
import json
import pickle
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

class APIReanalyzer:
    def __init__(self, api_choice="claude", api_key=None):
        self.api_choice = api_choice
        
        if api_choice == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif api_choice == "openai":
            self.client = OpenAI(api_key=api_key)
    
    def create_reanalysis_prompt(self, failed_result, ablation_data):
        """Create a prompt that includes the previous attempt for context."""
        
        prompt = f"""A previous analysis of this neural network latent produced low confidence results. Please provide a better interpretation.

## Previous Analysis
**Previous Hypothesis**: {failed_result.get('hypothesis', 'None provided')}
**Previous Confidence**: {failed_result.get('confidence', 'low')}
**Issues**: Insufficient evidence and unclear interpretation

## Latent Information
- Module: {failed_result['module_name']}
- Latent Index: {failed_result['latent_idx']}
- Effect Rate: {failed_result['effect_rate']:.1%} (percentage of examples where disabling this latent changed output)

## Examples Where This Latent Causally Affects Output

I'll show you several examples where this latent activated and changed the model's behavior when disabled:

"""
        # Add examples from ablation data
        # ... (use the same format as before but with more examples)
        
        prompt += """

Please provide a more thorough analysis considering:
1. The pattern across ALL examples shown
2. Both semantic and syntactic patterns
3. The specific way outputs change when this latent is disabled
4. Any subtle patterns the previous analysis might have missed

Provide your analysis in the same structured format as before."""
        
        return prompt
    
    async def analyze_batch_async(self, candidates, ablation_results_path, batch_size=5):
        """Process candidates in parallel batches."""
        
        # Load ablation data
        with open(ablation_results_path, 'rb') as f:
            ablation_results = pickle.load(f)
        
        results = []
        
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]
            batch_tasks = []
            
            for candidate in batch:
                # Get ablation data for this latent
                latent_data = ablation_results[candidate['module_name']][candidate['latent_idx']]
                prompt = self.create_reanalysis_prompt(candidate, latent_data)
                
                if self.api_choice == "claude":
                    task = self.call_claude(prompt)
                else:
                    task = self.call_openai(prompt)
                
                batch_tasks.append(task)
            
            # Wait for batch to complete
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Rate limiting
            time.sleep(1)
            
            if (i + batch_size) % 20 == 0:
                print(f"Processed {i + batch_size}/{len(candidates)} latents...")
        
        return results
    
    def call_claude(self, prompt):
        """Synchronous Claude API call."""
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",  # Use Haiku for cost efficiency
            max_tokens=1024,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def call_openai(self, prompt):
        """Synchronous OpenAI API call."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content

if __name__=='__main__':

    # Usage
    reanalyzer = APIReanalyzer(api_choice="openai", api_key=os.environ.get("OPENAI_API_KEY"))

    with open('/home/cvenhoff/lora_interp/cache/ablation_results_r1024_k8_steps5000_batches2500_maxlen25.pkl', 'rb') as f:
        ablation_data = pickle.load(f)

    # Process top priority candidates
    with open('/home/cvenhoff/small_results/latents_for_api_review.json', 'r') as f:
        top_candidates = json.load(f)

    # Run reanalysis
    improved_results = []
    for candidate in tqdm(top_candidates[:10], desc="Reanalyzing with API"):
        prompt = reanalyzer.create_reanalysis_prompt(candidate, ablation_data)
        result = reanalyzer.call_openai(prompt)
        improved_results.append({
            **candidate,
            'improved_hypothesis': result,
            'api_model': 'gpt-4o'
        })

    with open('/home/cvenhoff/small_results/api_review.json', 'w+') as f:
        json.dump(improved_results, f)