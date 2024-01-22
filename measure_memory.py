# Script to measure memory usage and inference speed at high batch and sequence lengths.
# For comparison between MHA, MQA, GQA, and MLKV.

import subprocess
import re
import time
import torch
import threading
import pickle
from fire import Fire
from transformers import AutoTokenizer, set_seed

def generate_outputs(model, input_ids, attention_mask, sequence_length, results):
    # Store the generated outputs in the provided dictionary
    results['outputs'] = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=sequence_length)

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
    batch_size=128,
    sequence_length=2048,
):
    set_seed(69)
    # measure backgroud memory usage
    back_mem_usage = measure_memory()
    print(f"Background memory usage: {back_mem_usage}MB")

    if mlkv:
        from gpt_neox_mlkv import GPTNeoXForCausalLM
    else:
        from gpt_neox import GPTNeoXForCausalLM
    model = GPTNeoXForCausalLM.from_pretrained(model_name)
    model.to("cuda")
    # measure memory usage after loading model
    model_mem_usage = measure_memory() - back_mem_usage
    print(f"Model memory usage: {model_mem_usage}MB")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # create dummy input of batch_size x sequence_length//2 using random tokens from vocab
    max_token = len(tokenizer)
    input_ids = torch.randint(max_token, (batch_size, sequence_length//2), device="cuda")
    attention_mask = torch.ones_like(input_ids, device="cuda")

    print(f"Measuring memory usage and inference speed...")
    # time inference
    start = time.time()
    # generate outputs while measuring memory usage asynchonously
    # generation is just model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=sequence_length)
    # while that is running, measure memory usage like above

    # Before doing full generation, we do an empty run to measure the actual memory usage of the model + inputs
    # dump = {}
    # # init_thread = threading.Thread(target=generate_outputs, args=(model, torch.randint(1, (1, 1), device="cuda"), torch.ones(1, 1, device="cuda"), 1, dump))
    # init_thread = threading.Thread(target=generate_outputs, args=(model, input_ids, attention_mask, 1, dump))
    # init_thread.start()
    # while init_thread.is_alive():
    #     init_mem_usage = measure_memory()
    # init_thread.join()
    # print(f"Measured initial memory usage: {init_mem_usage}MB")

    # Dictionary to store the results of the generation
    generation_results = {}

    # Start the generation in a separate thread
    generation_thread = threading.Thread(target=generate_outputs, args=(model, input_ids, attention_mask, sequence_length, generation_results))
    generation_thread.start()

    # Measure memory usage while the generation is running
    while generation_thread.is_alive():
        # Same procedure to measure memory as before
        mem_usage = measure_memory()

    # Wait for the generation thread to finish
    generation_thread.join()

    print(f"Measured generation memory usage: {mem_usage}MB")
    # stop timer
    end = time.time()
    # compute memory usage
    mem_usage = mem_usage - back_mem_usage - model_mem_usage - init_mem_usage
    # compute inference time
    inference_time = end - start
    # compute inference speed
    inference_speed = ((sequence_length//2) * batch_size) / inference_time
    # print results
    print(f"Memory usage while inference: {mem_usage}MB")
    print(f"Inference time: {inference_time}s")
    print(f"Inference speed: {inference_speed}tokens/s")

if __name__ == "__main__":
    # torch.cuda.memory._record_memory_history(enabled=True)
    Fire(main)
    # s = torch.cuda.memory._snapshot()
    # with open('cuda_snapshot.pickle', "wb") as f:
    #     pickle.dump(s, f)
    # torch.cuda.memory._save_memory_usage("cuda_memory_usage.html")