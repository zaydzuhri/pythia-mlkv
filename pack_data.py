# Do packing on the data zaydzuhri/the_pile_tokenized_5percent_truncated
# In order to fill each batch up to the maximum length, we concatenate texts together with EOS token as separator
# Then upload as a hf dataset zaydzuhri/the_pile_tokenized_5percent_truncated_packed

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from fire import Fire

def main(
    dataset_name: str = "zaydzuhri/the_pile_tokenized_5percent_truncated",
    sequence_length: int = 2048,
):
    # Initialize tokenizer with a specific model
    tokenizer = AutoTokenizer.from_pretrained("../pythia-160m-deduped")  # Adjusted to use GPT-2's tokenizer as an example
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set to EOS token

    dataset = load_dataset(dataset_name, split='train')

    def pack_batch(batch):
        # EVEN NEWER APPROACH: if using the above strategy, the packing isn't optimal
        # Instead, we should pack as many documents as possible into a single pack
        # So we look at every document length in the batch, and find the combination of documents that fit
        # Probably the simplest way is to first pack the shortest documents together, and then the longest
        # Basically just reorder the batch by document length, and run the previous strategy
        reordered_batch = sorted(batch['input_ids'], key=lambda x: len(x))
        packed_ids = []
        buffer_ids = []
        for ids in reordered_batch:
            ids_with_eos = ids + [tokenizer.eos_token_id]
            new_length = len(buffer_ids) + len(ids_with_eos)
            if new_length <= sequence_length:
                buffer_ids.extend(ids_with_eos)
            else:
                packed_ids.append(buffer_ids)
                buffer_ids = ids_with_eos
        packed_ids.append(buffer_ids)
                
        return {'input_ids': packed_ids}

    packed_dataset = dataset.map(
        pack_batch,
        batched=True,
        batch_size=2000,
        remove_columns=dataset.column_names
    )

    # # Quick check to confirm all rows are filled to sequence_length
    # def check_full_length(sample):
    #     for i, row in enumerate(sample):
    #         if len(row['input_ids']) != sequence_length:
    #             print(f"Row {i} does not meet the sequence length requirement.")
    #             return False
    #     print(f"All checked rows meet the sequence length of {sequence_length}.")
    #     return True

    # try:
    #     # use the entire dataset for a thorough check
    #     check_full_length(packed_dataset)
    # except:
    #     pass
    
    # Make sure to authenticate on Hugging Face before pushing
    packed_dataset.push_to_hub('zaydzuhri/the_pile_tokenized_5percent_truncated_packed_v2')

# To enable running this script from the command line
if __name__ == "__main__":
    import fire
    fire.Fire(main)

    