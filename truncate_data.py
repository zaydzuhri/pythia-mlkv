from datasets import load_dataset, DatasetDict
import copy

train_dataset = load_dataset('zaydzuhri/the_pile_tokenized_5percent')
if isinstance(train_dataset, DatasetDict):
    train_dataset = train_dataset["train"]

def truncate(sample):
    sample["input_ids"] = sample["input_ids"][0:2048]
    # sample["attention_mask"] = sample["attention_mask"][0:2048]
    return sample

train_dataset = train_dataset.map(
    truncate, desc="Truncating")

# drop attention_mask column


# train_dataset.save_to_disk('/mnt/h/server/the_pile_tokenized_5percent_truncated') 
# to disk doesn't work, upload to hub instead
train_dataset.push_to_hub('zaydzuhri/the_pile_tokenized_5percent_truncated')