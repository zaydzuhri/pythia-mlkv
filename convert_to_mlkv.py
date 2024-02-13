# Convert existing HF EleutherAI Pythia weights to MQA or GQA or MLKV 
# given two new params:
# num_key_value_heads: number of kv heads in a layer (MQA=1, GQA>1)
# num_key_value_layers: number of layers that have kv heads (MLKV)
# Merge KV heads by averaging them

import torch
import json
import os
import shutil
from fire import Fire

def main(
    weights_path: str,
    num_key_value_layers: int,
    num_key_value_heads: int,
    output_path: str = None,
):
    if output_path is None:
        output_path = weights_path + "_mlkv_" + str(num_key_value_layers) + "_" + str(num_key_value_heads)
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    config = json.load(open(weights_path+"/config.json"))
    # add new params
    config["num_key_value_heads"] = num_key_value_heads
    config["num_key_value_layers"] = num_key_value_layers
    # load old weights
    state_dict = torch.load(weights_path+"/pytorch_model.bin", map_location="cpu")
    # not only do we need to merge the kv heads,
    # we also need to seperate the query_key_value projections into seperate projections query, key, value
    # let's do that first
    head_size = config["hidden_size"] // config["num_attention_heads"]
    num_layers = config["num_hidden_layers"]
    for i in range(num_layers):
        # get the weights
        query_key_value_weight = state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
        query_key_value_bias = state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
        # need to transpose them
        query_key_value_weight = query_key_value_weight
        # seperate them
        # this is a bit tricky. the weights are of shape [num_attention_heads*3*hidden_size, hidden_size]
        # we need to split them into 3 parts of shape [num_attention_heads*head_size, hidden_size]
        # as you can see, the 3-part QKV is not the first primary dimension in the flattened weights, instead it's the second after num_attention_heads
        # so we need to reshape the weights to [num_attention_heads, 3*head_size, hidden_size] first
        query_key_value_weight = query_key_value_weight.view(config["num_attention_heads"], 3, head_size, config["hidden_size"])
        # then we can split them
        query_weight = query_key_value_weight[:,0,:,:].reshape(config["num_attention_heads"]*head_size, config["hidden_size"])
        key_weight = query_key_value_weight[:,1,:,:].reshape(config["num_attention_heads"]*head_size, config["hidden_size"])
        value_weight = query_key_value_weight[:,2,:,:].reshape(config["num_attention_heads"]*head_size, config["hidden_size"])
        # same for the biases
        query_key_value_bias = query_key_value_bias
        query_key_value_bias = query_key_value_bias.view(config["num_attention_heads"], 3, head_size)
        query_bias = query_key_value_bias[:,0,:].reshape(config["num_attention_heads"]*head_size)
        key_bias = query_key_value_bias[:,1,:].reshape(config["num_attention_heads"]*head_size)
        value_bias = query_key_value_bias[:,2,:].reshape(config["num_attention_heads"]*head_size)
        query_key_value_weight = query_key_value_weight.view(config["hidden_size"], config["num_attention_heads"], 3*head_size)
        # save them with transposed shape
        state_dict[f"gpt_neox.layers.{i}.attention.query.weight"] = query_weight
        state_dict[f"gpt_neox.layers.{i}.attention.query.bias"] = query_bias
        state_dict[f"gpt_neox.layers.{i}.attention.key.weight"] = key_weight
        state_dict[f"gpt_neox.layers.{i}.attention.key.bias"] = key_bias
        state_dict[f"gpt_neox.layers.{i}.attention.value.weight"] = value_weight
        state_dict[f"gpt_neox.layers.{i}.attention.value.bias"] = value_bias
        # delete the old weights
        del state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.weight"]
        del state_dict[f"gpt_neox.layers.{i}.attention.query_key_value.bias"]
    # now we can merge the kv heads
    # for same-layer kv sharing, just average the weights to the number of kv heads
    # for multi-layer kv sharing, need to only keep the kv heads in the specified layers and average the others into them
    # let's do same-layer kv sharing
    # skip if num_key_value_heads == num_attention_heads
    if num_key_value_heads == config["num_attention_heads"]:
        print("num_key_value_heads == num_attention_heads, skipping same-layer kv sharing")
    else:
        for i in range(num_layers):
            # get the weights
            key_weight = state_dict[f"gpt_neox.layers.{i}.attention.key.weight"]
            key_bias = state_dict[f"gpt_neox.layers.{i}.attention.key.bias"]
            value_weight = state_dict[f"gpt_neox.layers.{i}.attention.value.weight"]
            value_bias = state_dict[f"gpt_neox.layers.{i}.attention.value.bias"]
            # those have shape [num_heads*head_size, hidden_size]
            # need to get them to [num_key_value_heads*head_size, hidden_size, ] where num_key_value_heads < num_heads
            # we can just average the weights
            key_weight = key_weight.view(config["num_attention_heads"], head_size, config["hidden_size"])
            key_bias = key_bias.view(config["num_attention_heads"], head_size)
            value_weight = value_weight.view(config["num_attention_heads"], head_size, config["hidden_size"])
            value_bias = value_bias.view(config["num_attention_heads"], head_size)
            # average the groups of heads
            group_size = config["num_attention_heads"] // config["num_key_value_heads"]
            new_key_weight = torch.zeros(config["num_key_value_heads"], head_size, config["hidden_size"])
            new_key_bias = torch.zeros(config["num_key_value_heads"], head_size)
            new_value_weight = torch.zeros(config["num_key_value_heads"], head_size, config["hidden_size"])
            new_value_bias = torch.zeros(config["num_key_value_heads"], head_size)
            for j in range(config["num_key_value_heads"]):
                new_key_weight[j,:,:] = torch.mean(key_weight[j*group_size:(j+1)*group_size,:,:], dim=0)
                new_key_bias[j,:] = torch.mean(key_bias[j*group_size:(j+1)*group_size,:], dim=0)
                new_value_weight[j,:,:] = torch.mean(value_weight[j*group_size:(j+1)*group_size,:,:], dim=0)
                new_value_bias[j,:] = torch.mean(value_bias[j*group_size:(j+1)*group_size,:], dim=0)
            key_weight = new_key_weight
            key_bias = new_key_bias
            value_weight = new_value_weight
            value_bias = new_value_bias
            # reshape back to [hidden_size, head_size*num_key_value_heads]
            key_weight = key_weight.view(config["num_key_value_heads"]*head_size, config["hidden_size"])
            key_bias = key_bias.view(config["num_key_value_heads"]*head_size)
            value_weight = value_weight.view(config["num_key_value_heads"]*head_size, config["hidden_size"])
            value_bias = value_bias.view(config["num_key_value_heads"]*head_size)
            # save them
            state_dict[f"gpt_neox.layers.{i}.attention.key.weight"] = key_weight
            state_dict[f"gpt_neox.layers.{i}.attention.key.bias"] = key_bias
            state_dict[f"gpt_neox.layers.{i}.attention.value.weight"] = value_weight
            state_dict[f"gpt_neox.layers.{i}.attention.value.bias"] = value_bias
    # now multi-layer kv sharing
    # we need to average the weights of the layers that are not kv layers into the kv layers
    # skip if num_key_value_layers == num_hidden_layers
    if num_key_value_layers == config["num_hidden_layers"]:
        print("num_key_value_layers == num_hidden_layers, skipping multi-layer kv sharing")
    else:
        key_value_layers = [0] + [
                int((i + 1) * (config['num_hidden_layers'] / config['num_key_value_layers']))
                for i in range(config['num_key_value_layers'] - 1)
            ]
        for i in range(num_layers):
            if i not in key_value_layers:
                # get the weights
                key_weight = state_dict[f"gpt_neox.layers.{i}.attention.key.weight"]
                key_bias = state_dict[f"gpt_neox.layers.{i}.attention.key.bias"]
                value_weight = state_dict[f"gpt_neox.layers.{i}.attention.value.weight"]
                value_bias = state_dict[f"gpt_neox.layers.{i}.attention.value.bias"]
                # transpose them
                key_weight = key_weight
                value_weight = value_weight
                # average them to the previous closest kv layer
                prev_kv_layer = max([j for j in key_value_layers if j < i])
                prev_kv_key_weight = state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.key.weight"]
                prev_kv_key_bias = state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.key.bias"]
                prev_kv_value_weight = state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.value.weight"]
                prev_kv_value_bias = state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.value.bias"]
                key_weight = torch.mean(torch.stack([key_weight, prev_kv_key_weight]), dim=0)
                key_bias = torch.mean(torch.stack([key_bias, prev_kv_key_bias]), dim=0)
                value_weight = torch.mean(torch.stack([value_weight, prev_kv_value_weight]), dim=0)
                value_bias = torch.mean(torch.stack([value_bias, prev_kv_value_bias]), dim=0)
                # save them
                state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.key.weight"] = key_weight
                state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.key.bias"] = key_bias
                state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.value.weight"] = value_weight
                state_dict[f"gpt_neox.layers.{prev_kv_layer}.attention.value.bias"] = value_bias
                # delete the old weights
                del state_dict[f"gpt_neox.layers.{i}.attention.key.weight"]
                del state_dict[f"gpt_neox.layers.{i}.attention.key.bias"]
                del state_dict[f"gpt_neox.layers.{i}.attention.value.weight"]
                del state_dict[f"gpt_neox.layers.{i}.attention.value.bias"]
    # lastly we need to account for the lost parameters by beefing up the MLPs in all layers
    # calculate the total number of parameters lost in conversion first
    print("total heads before:", config['num_attention_heads'] * config['num_hidden_layers'])
    print("total heads after:", config['num_key_value_heads'] * config['num_key_value_layers'])
    num_heads_lost = (config['num_attention_heads'] * config['num_hidden_layers']) - (config['num_key_value_heads'] * config['num_key_value_layers'])
    print("num_heads_lost:", num_heads_lost)
    num_params_lost = num_heads_lost * (config['hidden_size'] * head_size * 2 + head_size * 2)
    print("num_params_lost:", num_params_lost)
    num_params_needed_layer = num_params_lost // config['num_hidden_layers']
    # now we need to upsize the intermediate layers of the MLPs by some amount
    # calculate the closest multiple of hidden_size that is nearest to the number of parameters lost
    # this is the amount we need to increase the intermediate size by
    intermediate_addition = int(round(num_params_needed_layer / config['hidden_size']))
    # add the intermediate_addition to the intermediate_size
    old_intermediate_size = config['intermediate_size']
    config['intermediate_size'] += intermediate_addition // 2 # there are 2 layers in the MLP
    print("intermediate_size increased by", intermediate_addition // 2)
    # initialize larger intermediate weights and biases, then put the old weights and biases in a slice of them
    if intermediate_addition == 0:
        print("intermediate_addition == 0, skipping intermediate layer upsizing")
    else:
        for i in range(num_layers):
            # get the weights
            intermediate_weight_in = state_dict[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"]
            intermediate_weight_out = state_dict[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"]
            intermediate_bias_in = state_dict[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias"]
            # initialize new weights and biases
            intermediate_weight_in_new = torch.zeros(config['intermediate_size'], config['hidden_size'])
            intermediate_weight_out_new = torch.zeros(config['hidden_size'], config['intermediate_size'])
            intermediate_bias_in_new = torch.zeros(config['intermediate_size'])
            # put the old weights and biases in the new ones
            intermediate_weight_in_new[:old_intermediate_size,:] = intermediate_weight_in
            intermediate_weight_out_new[:,:old_intermediate_size] = intermediate_weight_out
            intermediate_bias_in_new[:old_intermediate_size] = intermediate_bias_in
            # fill zeros with some part of the old weights and biases
            intermediate_weight_in_new[old_intermediate_size:, :] = intermediate_weight_in[:(config['intermediate_size']-old_intermediate_size), :]
            intermediate_weight_out_new[:, old_intermediate_size:] = intermediate_weight_out[:, :(config['intermediate_size']-old_intermediate_size)]
            intermediate_bias_in_new[old_intermediate_size:] = intermediate_bias_in[:(config['intermediate_size']-old_intermediate_size)]
            # save them
            state_dict[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.weight"] = intermediate_weight_in_new
            state_dict[f"gpt_neox.layers.{i}.mlp.dense_4h_to_h.weight"] = intermediate_weight_out_new
            state_dict[f"gpt_neox.layers.{i}.mlp.dense_h_to_4h.bias"] = intermediate_bias_in_new

    # save the new weights
    torch.save(state_dict, output_path+"/pytorch_model.bin")
    # save new config
    json.dump(config, open(output_path+"/config.json", "w"))
    # copy all the other files
    files = os.listdir(weights_path)
    for file in files:
        if file not in ["config.json", "pytorch_model.bin"]:
            shutil.copyfile(weights_path+"/"+file, output_path+"/"+file)

if __name__ == "__main__":
    Fire(main)