from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import apply_chat_template
from datasets import load_dataset
from peft import PeftModel


def get_conversational_dataset(dataset_name, tokenizer):
    dataset = load_dataset(dataset_name)

    if dataset_name == 'xlangai/spider':
        return get_spider_dataset(dataset_name, tokenizer)

    train, test = dataset['train'], dataset['test']

    train_dataset = train.map(
        apply_chat_template, fn_kwargs={'tokenizer': tokenizer}
    )
    test_dataset = test.map(
        apply_chat_template, fn_kwargs={'tokenizer': tokenizer}
    )

    return train_dataset, test_dataset


def get_spider_dataset(dataset_name, tokenizer):
    # source: https://medium.com/%40shekhars271991/finetuning-llama-3-2-eef3114b5f6c
    def format_entries(entry):
        # Format conversations as list of dictionaries with alternating user/assistant messages
        conversations = [
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": query}
            ]
            for question, query in zip(entry["question"], entry["query"])
        ]
        # Apply chat template to each conversation
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False)
            for convo in conversations
        ]
        return {"text": texts}

    train = load_dataset(dataset_name, split='train')
    test = load_dataset(dataset_name, split='validation[:10%]')
    train_dataset = train.map(format_entries, batched=True)
    test_dataset = test.map(format_entries, batched=True)
    return train_dataset, test_dataset


def merge_lora_adapter(
    base_model_dir: str,
    lora_checkpoint_dir: str,
    merged_output_dir: str,
    tokenizer_dir: str = None,
    torch_dtype="auto",
    device_map="auto"
):
    """
    Load a base model and its tokenizer (optionally from a separate directory),
    merge LoRA adapter weights, and save the merged model to the specified output directory.

    Args:
        base_model_dir (str): Path to the directory containing the base model files.
        lora_checkpoint_dir (str): Path to the directory containing LoRA adapter files 
                                   (e.g., adapter_config.json, lora_adapters.pt).
        merged_output_dir (str): Path to the directory where the merged model will be saved.
        tokenizer_dir (str, optional): Path to the directory from which to load the tokenizer.
                                       If None, defaults to `base_model_dir`.
        torch_dtype (str or torch.dtype, optional): Data type to load the model with. 
                                                   Defaults to 'auto'.
        device_map (str or dict, optional): Device map for loading the model. Defaults to 'auto'.

    Returns:
        None
    """
    # If no separate tokenizer directory is given, use the base_model_dir
    if tokenizer_dir is None:
        tokenizer_dir = base_model_dir

    # 1. Load the tokenizer from the specified directory
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # 2. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch_dtype,
        device_map=device_map
    )

    # 3. Load the LoRA adapter on top of the base model
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        lora_checkpoint_dir
    )

    # 4. Merge LoRA weights into the base model
    merged_model = model_with_lora.merge_and_unload()

    # 5. Save the merged model and tokenizer
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    print(f"Merged model saved to: {merged_output_dir}")
