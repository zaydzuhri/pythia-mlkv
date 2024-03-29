import os
import argparse
import copy
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from datetime import timedelta
from transformers import set_seed, DataCollatorForLanguageModeling, AutoTokenizer, Trainer, TrainingArguments

# from gpt_neox_mlkv import GPTNeoXForCausalLM, GPTNeoXConfig
# from gpt_neox import GPTNeoXForCausalLM, GPTNeoXConfig

def main(args):
    if 'mlkv' in args.model:
        from gpt_neox_mlkv import GPTNeoXForCausalLM, GPTNeoXConfig
    else:
        from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb
        wandb.login()
        wandb.init(project="mlkv", name=args.wandb)

    set_seed(args.seed)

    config_cls = GPTNeoXConfig
    model_cls = GPTNeoXForCausalLM

    config = config_cls.from_pretrained(args.model)

    model = model_cls.from_pretrained(
        args.model,
        config=config
    )

    if args.freeze_embeds:
        model.gpt_neox.embed_in.weight.requires_grad = False
        model.embed_out.weight.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = "<|padding|>"
    tokenizer.pad_token_id = 1
    # tokenizer.padding_side = "left"

    try:
        train_dataset = load_dataset(args.dataset)
    except:
        train_dataset = load_from_disk(args.dataset)
    if isinstance(train_dataset, DatasetDict):
        train_dataset = train_dataset["train"]

    if "input_ids" not in train_dataset.column_names:
        raise RuntimeError("Dataset must include an `input_ids` feature")
    if "attention_mask" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns('attention_mask')
    
    if args.truncate:
        def truncate(sample):
            sample["input_ids"] = sample["input_ids"][0:args.truncate]
            # sample["attention_mask"] = sample["attention_mask"][0:args.truncate]
            return sample
        train_dataset = train_dataset.map(
            truncate, desc="Truncating", num_proc=args.num_proc)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_steps = args.max_train_steps if args.max_train_steps != -1 else len(train_dataset) // (args.gradient_accumulate_every * args.batch_size)
    
    trainer_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_train_steps,
        gradient_accumulation_steps=args.gradient_accumulate_every,
        # per_device_eval_batch_size=32,
        # evaluation_strategy="steps",
        # eval_steps=5_000,
        logging_steps=5,
        save_steps=train_steps//5,
        num_train_epochs=1,
        weight_decay=0.01,
        # warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_schedule,
        learning_rate=args.learning_rate,
        # gradient_checkpointing=True,
        # save_steps=args.checkpointing_steps,
        # bf16=True,
        fp16=True,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        # push_to_hub=True,
        report_to="wandb" if args.wandb else "none",
        # report_to="none",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=trainer_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        # eval_dataset=None,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    trainer.push_to_hub()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=16)
    args.add_argument("--gradient-accumulate-every", type=int, default=4)
    args.add_argument("--resume-from-checkpoint", action="store_true")
    args.add_argument("--checkpointing-steps", type=int)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=-1)
    args.add_argument("--warmup-steps", type=int, default=100) # use ratio instead
    args.add_argument("--warmup-ratio", type=float, default=0.01)
    args.add_argument("--learning-rate", type=float, default=6e-4)
    args.add_argument("--grad-norm", action="store_true")
    args.add_argument("--model", type=str,
                      default="pythia-160m-deduped_mlkv")
    args.add_argument("--truncate", type=int, default=None)
    args.add_argument("--dataset", type=str,
                      default="zaydzuhri/the_pile_tokenized_5percent_truncated_packed_v2")
    args.add_argument("--deepspeed", action="store_true")
    args.add_argument("--num-proc", type=int, default=32)
    args.add_argument("--lr-schedule", type=str,
                      choices=["linear", "constant", "cosine"], default="cosine")
    args.add_argument("--save-only", action="store_true")
    args.add_argument("--log-loss", type=str)
    args.add_argument("--freeze-embeds", action="store_true")
    main(args.parse_args())