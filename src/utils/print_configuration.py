import textwrap
from tabulate import tabulate

def print_training_config(languages, nrows, model, tokenizer, model_params):

    # Extract key model configuration details
    model_info = {
        "model_name": model.config._name_or_path,
        "num_hidden_layers": model.config.num_hidden_layers,
        "hidden_size": model.config.hidden_size,
        "num_attention_heads": model.config.num_attention_heads,
        "num_labels": model.config.num_labels
    }

    # Extract key tokenizer details
    tokenizer_info = {
        "tokenizer_name": tokenizer.name_or_path,
        "vocab_size": tokenizer.vocab_size,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "cls_token_id": tokenizer.cls_token_id,
        "sep_token_id": tokenizer.sep_token_id
    }

    # Prepare the data for tabulation
    headers = ["Configuration Item", "Details"]
    config_data = [
        ["Languages", "\n".join(textwrap.wrap(", ".join(languages), width=40))],
        ["Number of Rows per Language", nrows],
        ["Model Info", "\n".join(f"{k}: {v}" for k, v in model_info.items())],
        ["Tokenizer Info", "\n".join(f"{k}: {v}" for k, v in tokenizer_info.items())],
        ["Model Parameters", "\n".join(f"{k}: {v}" for k, v in model_params.items())],
    ]

    # Print the configuration
    print("Training Loop Configuration\n")
    print(tabulate(config_data, headers=headers, tablefmt="grid"))
