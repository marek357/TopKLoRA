import glob
import json
from tqdm import tqdm
import os

THRESHOLD = 0.70 * 90


def find_cache_file(base_path, adapter_name, cache_file_map):
    if not adapter_name in cache_file_map:
        files = glob.glob(base_path + '/llm_cache*/**/*.json', recursive=True)
        for file in tqdm(files, disable=True):
            with open(file, 'r') as f:
                data = json.load(f)
                cache_file_map[data['latent']] = file

    if adapter_name in cache_file_map:
        return cache_file_map[adapter_name], cache_file_map

    return None, cache_file_map


if __name__ == '__main__':

    directories = glob.glob(
        '/scratch/network/ssd/marek/sparselora/autointerp/dpo_model_full*'
        # '/scratch/network/ssd/marek/sparselora/autointerp/dpo_model_18layer*'
    )

    try:
        with open('cache/cache_file_map.json', 'r') as f:
            cache_file_map = json.load(f)

        # update cache_file_map with any new directories
        for directory in directories:
            model_name = directory.split("/")[-1]
            if model_name not in cache_file_map:
                cache_file_map[model_name] = {}

    except Exception:
        cache_file_map = {
            directory.split("/")[-1]: {} for directory in directories
        }

    for directory in directories:
        print(f'Results for model: {directory.split("/")[-1]}')
        explanations = glob.glob(directory + '/explanations/*/*.json')
        scores = glob.glob(directory + '/scores/*/*.json')

        if len(scores) == 0:
            continue

        for file in scores:
            correct = 0
            with open(file, 'r') as f:
                data = json.load(f)
            for d in data:
                try:
                    local_correct = int(d['correct'])
                    correct += local_correct
                except Exception:
                    pass
            if correct > THRESHOLD:
                explanation = file.replace('enhanced_detection', 'enhanced_default').replace(
                    'scores', 'explanations')
                with open(explanation, 'r') as f:
                    expl = json.load(f)
                adapter_name = file.split("/")[-1].replace(
                    ".json", ""
                ).replace(
                    "base_model_model_model_layers_11_", ""
                ).replace(
                    "_", "-"
                )
                cache_adapter_name = 'base_model.model.model.layers.11.'

                if 'self-attn' in adapter_name:
                    cache_adapter_name += 'self_attn.'
                    matrix = 'self_attn ('
                    weight_matrix = adapter_name.split('-')[8]
                else:
                    cache_adapter_name += 'mlp.'
                    matrix = 'mlp ('
                    weight_matrix = adapter_name.split('-')[7]
                # print(adapter_name)
                # assert False

                latent_num = adapter_name.split('-')[-1]
                matrix += weight_matrix + \
                    f'_proj, {latent_num.removeprefix("latent")}))'
                cache_adapter_name += f'{weight_matrix}_proj.topk_'
                cache_adapter_name += adapter_name.split('-')[-1]
                # self_attn.q_proj.topk_latent217
                cache_file_model_map = cache_file_map[directory.split("/")[-1]]

                cache_file, cache_file_model_map = find_cache_file(
                    directory, cache_adapter_name, cache_file_model_map
                )

                cache_file_map[directory.split("/")[-1]] = cache_file_model_map

                with open('cache/cache_file_map.json', 'w+') as f:
                    json.dump(cache_file_map, f)

                print(
                    round(correct / 90, 2),
                    expl['explanation'],
                    f'Matrix: {matrix}\n',
                    # f'(Adapter: {cache_file})'
                )

        print('---')
