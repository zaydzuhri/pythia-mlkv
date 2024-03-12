# Convert existing HF EleutherAI Pythia weights to MQA or GQA or MLKV 
# given two new params:
# num_key_value_heads: number of kv heads in a layer (MQA=1, GQA>1)
# num_key_value_layers: number of layers that have kv heads (MLKV)
# Merge KV heads by averaging them

import torch
import json
import os
import math
import shutil
from fire import Fire
from gpt_neox_mlkv import GPTNeoXForCausalLM, GPTNeoXConfig

def main(
    weights_path: str,
    num_key_value_layers: int,
    num_key_value_heads: int,
    output_path: str = None,
    use_key_value_mlp: bool = False
):
    if output_path is None:
        output_path = weights_path + "_mlkv_" + ("mlp_" if use_key_value_mlp else "") + str(num_key_value_layers) + "_" + str(num_key_value_heads)
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
            # need to get them to [num_key_value_heads*head_size, hidden_size] where num_key_value_heads < num_heads
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
            # reshape back to [head_size*num_key_value_heads, hidden_size]
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
    if not use_key_value_mlp:
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
    else:
        config['use_key_value_mlp'] = True
        # If we're replacing linear key value projections with MLPs, we need to create new weights and biases for them
        # These MLPs will have the names gpt_neox.layers.{i}.attention.key.dense_in and gpt_neox.layers.{i}.attention.key.dense_out with weights and biases. same for value
        # initialize new weights and biases (with proper initiliazation because we might not be able to fill them with the old weights and biases)
        # but first we need to calculate the intermediate size of the new MLPs so that we can make up for the lost parameters
        num_params_needed_kv_layer = (num_params_lost // config['num_key_value_layers']) // 2 # there are key and value
        # we have one linear projection with size [num_key_value_heads*head_size, hidden_size] for key and value
        # the MLP will have 2 linear layers with size [mlp_intermediate_size, hidden_size] and [num_key_value_heads*head_size, mlp_intermediate_size] 
        # to simplify the calculation, add the already available parameters to the needed parameters so that we can count the MLP parameters from scratch
        num_params_needed_kv_layer += config['num_key_value_heads']*head_size*config['hidden_size']
        # for every 1 MLP intermediate size, we get hidden_size + (num_key_value_heads*head_size) parameters
        # get the closest multiple of that to the number of parameters needed
        multiplier = (config['hidden_size'] + (config['num_key_value_heads']*head_size))
        mlp_intermediate_size = int(round(num_params_needed_kv_layer / multiplier))
        print("KV MLP intermediate size:", mlp_intermediate_size)
        config['kv_intermediate_size'] = mlp_intermediate_size
        for i in key_value_layers:
            key_weight_in = torch.empty(mlp_intermediate_size, config['hidden_size'])
            key_weight_out = torch.empty(config["num_key_value_heads"]*head_size, mlp_intermediate_size)
            key_bias_in = torch.empty(mlp_intermediate_size).uniform_(-1, 1)
            key_bias_out = torch.empty(config["num_key_value_heads"]*head_size).uniform_(-1, 1)
            value_weight_in = torch.empty(mlp_intermediate_size, config['hidden_size'])
            value_weight_out = torch.empty(config["num_key_value_heads"]*head_size, mlp_intermediate_size)
            value_bias_in = torch.empty(mlp_intermediate_size).uniform_(-1, 1)
            value_bias_out = torch.empty(config["num_key_value_heads"]*head_size).uniform_(-1, 1)
            # Initilialize them with nn.init.kaiming_uniform_
            torch.nn.init.kaiming_uniform_(key_weight_in, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(key_weight_out, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(value_weight_in, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(value_weight_out, a=math.sqrt(5))
            # Put in the old weights in the first neurons of the first layer, just because we can
            key_weight_in[:config['num_key_value_heads']*head_size, :] = state_dict[f"gpt_neox.layers.{i}.attention.key.weight"]
            value_weight_in[:config['num_key_value_heads']*head_size, :] = state_dict[f"gpt_neox.layers.{i}.attention.value.weight"]
            key_bias_in[:config['num_key_value_heads']*head_size] = state_dict[f"gpt_neox.layers.{i}.attention.key.bias"]
            value_bias_in[:config['num_key_value_heads']*head_size] = state_dict[f"gpt_neox.layers.{i}.attention.value.bias"]
            # save them
            state_dict[f"gpt_neox.layers.{i}.attention.key.dense_in.weight"] = key_weight_in
            state_dict[f"gpt_neox.layers.{i}.attention.key.dense_out.weight"] = key_weight_out
            state_dict[f"gpt_neox.layers.{i}.attention.key.dense_in.bias"] = key_bias_in
            state_dict[f"gpt_neox.layers.{i}.attention.key.dense_out.bias"] = key_bias_out
            state_dict[f"gpt_neox.layers.{i}.attention.value.dense_in.weight"] = value_weight_in
            state_dict[f"gpt_neox.layers.{i}.attention.value.dense_out.weight"] = value_weight_out
            state_dict[f"gpt_neox.layers.{i}.attention.value.dense_in.bias"] = value_bias_in
            state_dict[f"gpt_neox.layers.{i}.attention.value.dense_out.bias"] = value_bias_out
            # delete the old linear weights
            del state_dict[f"gpt_neox.layers.{i}.attention.key.weight"]
            del state_dict[f"gpt_neox.layers.{i}.attention.key.bias"]
            del state_dict[f"gpt_neox.layers.{i}.attention.value.weight"]
            del state_dict[f"gpt_neox.layers.{i}.attention.value.bias"]

    # save the new weights
    torch.save(state_dict, output_path+"/pytorch_model.bin")
    # save new config
    json.dump(config, open(output_path+"/config.json", "w"))
    # copy all the other files
    files = os.listdir(weights_path)
    for file in files:
        if file not in ["config.json", "pytorch_model.bin"]:
            shutil.copyfile(weights_path+"/"+file, output_path+"/"+file)

    # check total number of parameters
    model = GPTNeoXForCausalLM.from_pretrained(output_path)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Final parameter count:", total_params)

if __name__ == "__main__":
    Fire(main)