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
        packed_ids = []
        buffer_ids = []
        buffer_length = 0

        for ids in batch['input_ids']:
            ids_with_eos = ids + [tokenizer.eos_token_id]  # Append EOS token at the end of each document
            while ids_with_eos:
                space_left = sequence_length - buffer_length
                # If the current document (or part of it) fits into the buffer
                if len(ids_with_eos) <= space_left:
                    buffer_ids.extend(ids_with_eos)
                    buffer_length += len(ids_with_eos)
                    ids_with_eos = []  # Clear current document since it has been fully added
                else:
                    # If not, add what fits, and keep the rest for the next batch
                    buffer_ids.extend(ids_with_eos[:space_left])
                    ids_with_eos = ids_with_eos[space_left:]  # Keep remainder for next
                    buffer_length = sequence_length
                
                # If buffer is full, flush it to packed_ids
                if buffer_length == sequence_length:
                    packed_ids.append(buffer_ids)
                    buffer_ids = []
                    buffer_length = 0

        return {'input_ids': packed_ids}

    packed_dataset = dataset.map(
        pack_batch,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Quick check to confirm all rows are filled to sequence_length
    def check_full_length(sample):
        for i, row in enumerate(sample):
            if len(row['input_ids']) != sequence_length:
                print(f"Row {i} does not meet the sequence length requirement.")
                return False
        print(f"All checked rows meet the sequence length of {sequence_length}.")
        return True

    try:
        # use the entire dataset for a thorough check
        check_full_length(packed_dataset)
    except:
        pass
    
    # Make sure to authenticate on Hugging Face before pushing
    packed_dataset.push_to_hub('zaydzuhri/the_pile_tokenized_5percent_truncated_packed_full')

# To enable running this script from the command line
if __name__ == "__main__":
    import fire
    fire.Fire(main)

    