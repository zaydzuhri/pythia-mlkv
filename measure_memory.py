# Script to measure memory usage and inference speed at high batch and sequence lengths.
# For comparison between MHA, MQA, GQA, and MLKV.

import subprocess
import sys
import os
import re
import time
import torch
import gc
import threading
import pickle
from fire import Fire
from transformers import AutoTokenizer, set_seed
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

def generate_outputs(model, input_ids, past_key_values, attention_mask, sequence_length, batch_size, results):
    # Measure memory usage while the generation is running
    mem_usage = []
    # Store the generated outputs in the provided dictionary
    # results['outputs'] = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=sequence_length)
    # HuggingFace generate has a lot of overhead, let's do manual generation using model.forward
    # But we need to keep the cache manually
    # sequence_length = 4
    # total_mem = 0
    n_generated_tokens = 0
    try:
        start = time.time()
        with torch.no_grad():
            for i in (bar := tqdm(range(sequence_length - 1))):
                # outputs = model.forward(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
                outputs = model.forward(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                n_generated_tokens += 1
                # print(f'{torch.cuda.memory_allocated() / 1024 / 1024}MB')
                # input_ids = None
                # input_ids = outputs.logits[:, -1].argmax(-1).unsqueeze(-1)
                # past_key_values = outputs.past_key_values
                input_ids = outputs[0][:, -1].argmax(-1).unsqueeze(-1)
                past_key_values = outputs[1]
                outputs = None
                torch.cuda.empty_cache()
                mem_usage.append(measure_memory())
                # print(f'\npast_key_values.dtype: {past_key_values[0][0].dtype}\nelement size: {past_key_values[0][0].element_size()}\n')
                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             print(type(obj), str(list(obj.shape)), obj.dtype, obj.device, obj.requires_grad)
                #             total_mem += obj.element_size() * obj.nelement() / 1024 / 1024
                #     except:
                #         pass
                # kv_cache_mem = past_key_values[0][0].element_size() * past_key_values[0][0].nelement()*len(past_key_values)*len(past_key_values[0]) / 1024 / 1024
                # bar.set_description(f"Generated {i+1}/{sequence_length//2}, Cache Memory: {kv_cache_mem:.2f}MB")
                elapsed = bar.format_dict["elapsed"]
                # Stop if more than 30 minutes
                if elapsed > 1800:
                    break
                # print(f"Memory usage: {mem_usage}MB")
                # print(f"{torch.cuda.memory_allocated()/1024/1024}MB")
        # print(f"Total Memory: {total_mem:.2f}MB")
        # mem_usage = measure_memory()
        end = time.time()
        inference_time = end - start
        inference_speed = (n_generated_tokens * batch_size) / inference_time
        # print(f"Inference time: {inference_time:.2f}s")
        # print(f"Inference speed: {inference_speed:.2f} tokens/s")
        kv_cache_mem = (past_key_values[0][0].element_size() * past_key_values[0][0].nelement()*len(past_key_values)*len(past_key_values[0])) / 1024 / 1024
        # For debugging, see characteristics of past_key_values and print part of it
        # print('past_key_values:', type(past_key_values), len(past_key_values), type(past_key_values[0]), len(past_key_values[0]), type(past_key_values[0][0]), past_key_values[0][0].shape, past_key_values[0][0].dtype, past_key_values[0][0].device, past_key_values[0][0].requires_grad)
        # print(past_key_values[0][0][:1])
        # print(f"Final KV Cache Memory: {kv_cache_mem:.2f}MB")
        # Mean memory usage
        results['mem_usage'] = sum(mem_usage) / len(mem_usage)
        results['inference_time'] = inference_time
        results['inference_speed'] = inference_speed
        results['kv_cache_mem'] = kv_cache_mem
        results['is_oom'] = False
    except RuntimeError as e:
        print(f"{e}\nSkipping batch size {batch_size}...")
        results['mem_usage'] = float("inf")
        results['inference_time'] = float("inf")
        results['inference_speed'] = 0
        results['kv_cache_mem'] = 0
        results['is_oom'] = True

def measure_memory():
    with subprocess.Popen(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"],
        stdout=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    ) as mem_proc:
        mem_proc.wait()
        mem_out = mem_proc.stdout.read()
        mem_usage = int(re.findall(r"\d+", mem_out)[0])
    return mem_usage

def main(
    model_name,
    mlkv=False,
    # batch_sizes=[8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96],
    batch_sizes=[8, 16, 24, 32, 40, 48, 64, 80, 100, 140, 220, 300, 380, 460, 620, 780, 940, 1100],
    sequence_length=48,
    prefill_length=2000,
    log_file=None,
    measure_loss=False,
    measure_profile=False,
    revision=None,
):
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    set_seed(69)
    # measure backgroud memory usage
    back_mem_usage = measure_memory()
    print(f"Background memory usage: {back_mem_usage}MB")

    if mlkv:
        from gpt_neox_mlkv import GPTNeoXForCausalLM
    else:
        from gpt_neox import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=revision)
    n_layers = model.config.num_hidden_layers if not mlkv else model.config.num_key_value_layers
    n_heads = model.config.num_attention_heads if not mlkv else model.config.num_key_value_heads
    model.to("cuda")
    # measure memory usage after loading model
    model_mem_usage = measure_memory() - back_mem_usage
    print(f"Model memory usage: {model_mem_usage}MB")

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
    for batch_size in batch_sizes:
        print(f"Measuring memory usage and inference speed for batch size {batch_size}...")
        # create dummy input of batch_size x sequence_length//2 using random tokens from vocab
        max_token = len(tokenizer)
        # input_ids = torch.randint(max_token, (batch_size, sequence_length//2), device="cuda")
        # past_key_values = None
        # Instead of doing a cold start, let's do a warm start with a single token and a randomly initialized past_key_values of length prefill_length
        # past_key_values size is (n_layers, 2, batch_size, n_heads, prefill_length, hidden_size//model.config.num_attention_heads) but the first two are tuples
        input_ids = torch.randint(max_token, (batch_size, 1), device="cuda")
        past_key_values = tuple(tuple(torch.randn(batch_size, n_heads, prefill_length, 64, device="cuda", dtype=torch.float16) for _ in range(2)) for _ in range(n_layers))
        # print(f'past_key_values.dtype: {past_key_values[0][0].dtype}\nelement size: {past_key_values[0][0].element_size()}\n')
        attention_mask = None

        if measure_profile:
            # profiling
            print(f"Profiling first forward pass...")
            past_key_values = None
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    outputs = model.forward(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
                    input_ids_next = outputs.logits[:, -1].argmax(-1).unsqueeze(-1)
                    past_key_values = outputs.past_key_values
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

            print(f"Profiling second forward pass with KV cache...")
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, use_cuda=True) as prof:
                with record_function("model_inference"):
                    outputs = model.forward(input_ids=input_ids_next, past_key_values=past_key_values, use_cache=True, return_dict=True)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        print(f"Measuring memory usage and inference speed...")
        # outputs = model.forward(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, return_dict=True)
        # Dictionary to store the results of the generation
        generation_results = {}

        # Start the generation in a separate thread
        # generation_thread = threading.Thread(target=generate_outputs, args=(model, input_ids, past_key_values, attention_mask, sequence_length, batch_size, generation_results))
        # generation_thread.start()
        generate_outputs(model, input_ids, past_key_values, attention_mask, sequence_length, batch_size, generation_results)

        # Measure memory usage while the generation is running
        # mem_usage = 0
        # while generation_thread.is_alive():
        #     # Same procedure to measure memory as before but only keep max value
        #     mem_usage = max(mem_usage, measure_memory())

        # Wait for the generation thread to finish
        # generation_thread.join()
        mem_usage = generation_results['mem_usage']
        inference_time = generation_results['inference_time']
        inference_speed = generation_results['inference_speed']
        kv_cache_mem = generation_results['kv_cache_mem']
        is_oom = generation_results['is_oom']

        if is_oom:
            print(f"Out of memory error occurred. Assuming all subsequent batch sizes will also be OOM. Exiting...")
            break

        print(f"Measured generation memory usage: {mem_usage}MB")
        # compute memory usage
        mem_usage = mem_usage - back_mem_usage - model_mem_usage
        # print results
        print(f"Memory usage while inference: {mem_usage}MB")
        print(f"Inference time: {inference_time:.2f}s")
        print(f"Inference speed: {inference_speed:.2f} tokens/s")
        print(f"KV Cache Memory: {kv_cache_mem:.2f}MB")

        if log_file:
            if not os.path.exists(log_file):
                with open(log_file, "w") as f:
                    f.write("model_name,batch_size,sequence_length,background_memory_usage,model_memory_usage,memory_usage,kv_cache_mem,inference_time,inference_speed,is_oom\n")
            # log file should be csv with columns: model_name, batch_size, sequence_length, memory_usage, inference_time, inference_speed
            with open(log_file, "a") as f:
                f.write(f"{model_name},{batch_size},{sequence_length},{back_mem_usage},{model_mem_usage},{mem_usage},{kv_cache_mem},{inference_time},{inference_speed},{is_oom}\n")

    if measure_loss:
        from datasets import load_dataset
        dataset = load_dataset("zaydzuhri/the_pile_tokenized_5percent_truncated_packed", split='train[:4]')
        # already tokenized, the only column is input_ids
        # add labels column so that the model outputs loss
        dataset = dataset.map(lambda x: {'labels': x['input_ids']}, batched=True)
        # model inputs must be tensors
        dataset.set_format(type='torch', columns=['input_ids', 'labels'])
        # measure loss. move input to cuda before passing to model
        outputs = model(input_ids=dataset['input_ids'].to("cuda"), labels=dataset['labels'].to("cuda"))
        print(f"Loss: {outputs.loss.item()}")

if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history(enabled=True)
    Fire(main)
    # s = torch.cuda.memory._snapshot()
    # with open('cuda_snapshot.pickle', "wb") as f:
    #     pickle.dump(s, f)
    # torch.cuda.memory._save_memory_usage("cuda_memory_usage.html")