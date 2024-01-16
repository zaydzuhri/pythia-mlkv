# For uptraining, we need ~5% of the original dataset
# Pythia uses The Pile (EleutherAI/the_pile_deduplicated), which is very large and untokenized
# We need to tokenize 5% of it and store in a hf dataset
# This script streams shuffled lines from the_pile_deduplicated and tokenizes them
# and makes a hf dataset and uploads it to hf

import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from functools import partial
from fire import Fire
from tqdm import tqdm

def main(
    dataset_name: str = "EleutherAI/the_pile_deduplicated",
    output_dir: str = "H:/server/",
    output_name: str = "the_pile_tokenized_5percent",
    tokenizer_name: str = "EleutherAI/pythia-160m-deduped",
    seed: int = 42,
):
    if not os.path.exists(output_dir+output_name):
        os.mkdir(output_dir+output_name)

    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.take(6000000)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"])
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )

    # save to normal hf dataset
    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    hf_dataset = Dataset.from_generator(partial(gen_from_iterable_dataset, tokenized_dataset), features=tokenized_dataset.features)
    # print(hf_dataset[0])
    hf_dataset.save_to_disk(output_dir+output_name)
    hf_dataset.push_to_hub('zaydzuhri/'+output_name)

if __name__ == "__main__":
    Fire(main)