import os
import sys
import itertools


# system parameter
device_name = 'gypsum-titanx'
memory = '25GB'

# model parameter
model_type = "vbcbox" #["vTE","CRIM","dot","cosine","poe","box","vbcbox"]
node_features = "free" # ["random", "llm", "free"]
data_path_list = ["data/edges.body_part.n.01.json", "data/edges.mammal.n.01.json", "data/edges.commodity.n.01.json", "data/edges.disease.n.01.json", "data/edges.location.n.01.json"]
holdout_list = [0.5]
learning_rate_list = [1e-2, 1e-3, 1e-4]
num_layer_list = [1, 2, 3]
num_shared_layer_list = [0]
output_dim_list = [128]
intersection_temp_list = [0.01, 0.1, 1.0]

params = list(itertools.product(*[data_path_list, holdout_list, learning_rate_list, num_layer_list, num_shared_layer_list, output_dim_list, intersection_temp_list]))
for param in params:
    data_path, holdout, lr, num_layers, num_shared_layers, output_dim, it = param
    command = f"srun --partition={device_name} --gres=gpu:1 --mem={memory} python src/main.py --model_type {model_type} --node_features {node_features} --path {data_path} --holdout {holdout} --lr {lr} --num_layers {num_layers} --num_shared_layers {num_shared_layers} --output_dim {output_dim} --it {it} &"
    os.system(command)
