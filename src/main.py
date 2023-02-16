import sys
import numpy as np
import torch
import argparse
from data_loader import *
from torch.utils.data import DataLoader
from model import NeuralDot, NeuralCosine, NeuralComplex, NeuralPOE, NeuralBox, NeuralVBCBox, vTE, CRIM, Dot, Cosine, Box, VBCBox 
from utils import *
from train import *

parser = argparse.ArgumentParser(description='Encoded box project')
parser.add_argument('--path', type=str, default="data/edges.animal.n.01.json")
parser.add_argument('--model_type', type=str, default='vTE', choices=['vTE','CRIM','dot','cosine','complex','poe','box','vbcbox'])
parser.add_argument('--node_features', type=str, default='llm', choices=['free','random', 'glove', 'llm'])
parser.add_argument('--opt', type=str, default='adam', choices=['sgd','adam'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--it', type=float, default=0.01, help="intersection temperature for box embeddings")
parser.add_argument('--vt', type=float, default=1.0, help="volume temperature for box embeddings")
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--bz', type=int, default=1024, help="batch size")
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--output_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--num_shared_layers', type=int, default=0)
parser.add_argument('--holdout', type=float, default=0.5, help="proportion of edges for validation")

args = parser.parse_args()

args.path = args.path.strip("/")
sys.stdout = open(f"logs/log.{args.path.split('/')[-1]}_{args.node_features}_{args.model_type}_opt_{args.opt}_lr_{args.lr}_dp_{args.dropout}_bz_{args.bz}_hd_{args.hidden_dim}_od_{args.output_dim}_num_layers_{args.num_layers}_num_shared_layers_{args.num_shared_layers}_it_{args.it}_vt_{args.vt}_holdout_{args.holdout}", "w")
sys.stderr = open(f"logs/err.{args.path.split('/')[-1]}_{args.node_features}_{args.model_type}_opt_{args.opt}_lr_{args.lr}_dp_{args.dropout}_bz_{args.bz}_hd_{args.hidden_dim}_od_{args.output_dim}_num_layers_{args.num_layers}_num_shared_layers_{args.num_shared_layers}_it_{args.it}_vt_{args.vt}_holdout_{args.holdout}", "w")

print(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
sys.stdout.flush()

test_paths = ["data/edges.body_part.n.01.json", "data/edges.mammal.n.01.json", "data/edges.location.n.01.json", "data/edges.commodity.n.01.json", "data/edges.disease.n.01.json"]

if args.node_features == "glove":
    word_embeddings = np.load(f"data/embedding.300d.npy")
elif args.node_features == "llm":
    word_embeddings = np.load(f"data/embedding.instructor.1536d.npy")[:,:768]
else:
    word_embeddings = np.load(f"data/embedding.random.128d.npy")


N = word_embeddings.shape[0]
feat_dim = word_embeddings.shape[1]

if args.node_features != "free":
    if args.model_type == "vTE":
        model = vTE(feat_dim).to(device)
    elif args.model_type == "CRIM":
        model = CRIM(feat_dim, args.hidden_dim).to(device)
    elif args.model_type == "dot":
        model = NeuralDot(feat_dim, args.output_dim, args.hidden_dim, args.dropout, args.num_layers, args.num_shared_layers).to(device)
    elif args.model_type == "cosine":
        model = NeuralCosine(feat_dim, args.output_dim, args.hidden_dim, args.dropout, args.num_layers, args.num_shared_layers).to(device)
    elif args.model_type == "complex":
        model = NeuralComplex(feat_dim, args.output_dim, args.hidden_dim, args.dropout, args.num_layers, args.num_shared_layers).to(device)
    elif args.model_type == "poe":
        model = NeuralPOE(feat_dim, args.output_dim, args.hidden_dim, args.dropout, args.num_layers).to(device)
    elif args.model_type == "box":
        model = NeuralBox(feat_dim, args.output_dim, args.hidden_dim, args.dropout, args.num_layers, args.num_shared_layers, args.vt, args.it).to(device)
    elif args.model_type == "vbcbox":
        model = NeuralVBCBox(feat_dim, args.output_dim, args.hidden_dim, args.dropout, args.num_layers, args.num_shared_layers, args.vt, args.it).to(device)
else:
    if args.model_type == "dot":
        model = Dot(args.output_dim, N).to(device)
    elif args.model_type == "cosine":
        model = Cosine(args.output_dim, N).to(device)
    elif args.model_type == "box":
        model = Box(args.output_dim, N, args.vt, args.it).to(device)
    elif args.model_type == "vbcbox":
        model = VBCBox(args.output_dim, N, args.vt, args.it).to(device)


if len(list(model.parameters())) > 0:
    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
loss_fn = BCEWithLogsLoss()
#loss_fn = torch.nn.CrossEntropyLoss()


average_best_train_result = 0
average_best_valid_result_list = [0] * len(test_paths)
average_valid_result_train_neg_list = [0] * len(test_paths)
average_valid_result_test_list = [0] * len(test_paths)
average_test_result_list = [0] * len(test_paths)
average_test_result_train_neg_list = [0] * len(test_paths)
average_test_result_test_list = [0] * len(test_paths)
num_runs = 1
for run in range(num_runs):
    model.reset_parameters()
    random_seed = 2023 * (run + 1)
    torch.manual_seed(random_seed)
    train_data = HypernymGraph(args.path, word_embeddings, holdout=args.holdout, random_seed=random_seed)
    train_edge_coverage = train_data.train_graph.number_of_edges() / train_data.graph.number_of_edges()
    train_node_coverage = train_data.train_graph.number_of_nodes() / train_data.graph.number_of_nodes()
    test_data_list = []
    for p in test_paths:
        if p.strip("/") != args.path.strip("/"):
            test_data_list.append(HypernymGraph(p, word_embeddings, cross_holdout=0.9, random_seed=random_seed))
        else:
            test_data_list.append(train_data)

    train_dataloader = DataLoader(train_data, batch_size=args.bz, shuffle=True)
    test_dataloader_list = [DataLoader(d, batch_size=args.bz, shuffle=False) for d in test_data_list]

    for i in range(len(test_paths)):
        valid_metrics, test_metrics = test(test_dataloader_list[i], model, device)
        if test_paths[i].strip("/") == args.path.strip("/"):
            print(f"Run {run}, At beginning ({test_paths[i].split('.')[1]}): (same domain, node: {train_node_coverage:.3f}, edge: {train_edge_coverage:.3f}) overfitting AP = {valid_metrics['AP']:.3f}, valid AP = {test_metrics['AP']:.3f}, valid AP (training negatives) = {test_metrics['AP_train_neg']:.3f}, valid AP (test) = {test_metrics['AP_test']:.3f}")
        else:
            print(f"Run {run}, At beginning ({test_paths[i].split('.')[1]}): (cross domain) valid AP = {valid_metrics['AP']:.3f}, test AP = {test_metrics['AP']:.3f}, test AP (training negatives) = {test_metrics['AP_train_neg']:.3f}, test AP (test) = {test_metrics['AP_test']:.3f}")
        
        sys.stdout.flush()

    epochs = 50
    if args.num_layers == 0 and model_type not in ["vTE", "CRIM"]:
        epochs = 0

    best_train_result = 0
    best_valid_result_list = [0] * len(test_paths)
    valid_result_train_neg_list = [0] * len(test_paths)
    valid_result_test_list = [0] * len(test_paths)
    test_result_list = [0] * len(test_paths)
    test_result_train_neg_list = [0] * len(test_paths)
    test_result_test_list = [0] * len(test_paths)

    for t in range(epochs):
        print(f"Run {run}, Epoch {t}\n-------------------------------")
        sys.stdout.flush()
        train(train_dataloader, model, loss_fn, optimizer, device)
        for i in range(len(test_paths)):
            valid_metrics, test_metrics = test(test_dataloader_list[i], model, device)
            if_increased = ""
            if test_paths[i].strip("/") == args.path.strip("/"):
                if valid_metrics["AP"] > best_train_result:
                    best_train_result = valid_metrics["AP"]
                if test_metrics["AP"] > best_valid_result_list[i]:
                    best_valid_result_list[i] = test_metrics["AP"]
                    valid_result_train_neg_list[i] = test_metrics["AP_train_neg"]
                    valid_result_test_list[i] = test_metrics["AP_test"]
                    if_increased = "increased"
                print(f"Run {run}, At epoch {t} ({test_paths[i].split('.')[1]}): (same domain, node: {train_node_coverage:.3f}, edge: {train_edge_coverage:.3f}) overfitting AP = {valid_metrics['AP']:.3f}, {if_increased} valid AP = {test_metrics['AP']:.3f}, valid AP (training negatives) = {test_metrics['AP_train_neg']:.3f}, valid AP (test) = {test_metrics['AP_test']:.3f}")
            else:
                if valid_metrics["AP"] > best_valid_result_list[i]:
                    if_increased = "increased"
                    best_valid_result_list[i] = valid_metrics["AP"]
                    test_result_list[i] = test_metrics["AP"]
                    test_result_train_neg_list[i] = test_metrics["AP_train_neg"]
                    test_result_test_list[i] = test_metrics["AP_test"]
                print(f"Run {run}, At epoch {t} ({test_paths[i].split('.')[1]}): (cross domain) {if_increased} valid AP = {valid_metrics['AP']:.3f}, test AP = {test_metrics['AP']:.3f}, test AP (training negatives) = {test_metrics['AP_train_neg']:.3f}, test AP (test) = {test_metrics['AP_test']:.3f}")
            sys.stdout.flush()

    for i in range(len(test_paths)):
        if test_paths[i].strip("/") == args.path.strip("/"):
            print(f"Run {run}, Final ({test_paths[i].split('.')[1]}): (same domain, node: {train_node_coverage:.3f}, edge: {train_edge_coverage:.3f}) overfitting AP = {best_train_result:.3f}, best valid AP = {best_valid_result_list[i]:.3f}, valid AP (training negatives) = {valid_result_train_neg_list[i]:.3f}, valid AP (test) = {valid_result_test_list[i]:.3f}")
        else:
            print(f"Run {run}, Final ({test_paths[i].split('.')[1]}): (cross domain, node: {train_node_coverage:.3f}, edge: {train_edge_coverage:.3f}) best valid AP = {best_valid_result_list[i]:.3f}, test AP = {test_result_list[i]:.3f}, test AP (training negatives) = {test_result_train_neg_list[i]:.3f}, test AP (test) = {test_result_test_list[i]:.3f}")
    average_best_train_result += best_train_result
    for i in range(len(best_valid_result_list)):
        average_best_valid_result_list[i] += best_valid_result_list[i]
        average_valid_result_train_neg_list[i] += valid_result_train_neg_list[i]
        average_valid_result_test_list[i] += valid_result_test_list[i]
        average_test_result_list[i] += test_result_list[i]
        average_test_result_train_neg_list[i] += test_result_train_neg_list[i]
        average_test_result_test_list[i] += test_result_test_list[i]

    
average_best_train_result = average_best_train_result / num_runs
for i in range(len(best_valid_result_list)):
    average_best_valid_result_list[i] /= num_runs
    average_valid_result_train_neg_list[i] /= num_runs
    average_valid_result_test_list[i] /= num_runs
    average_test_result_list[i] /= num_runs
    average_test_result_train_neg_list[i] /= num_runs
    average_test_result_test_list[i] /= num_runs

for i in range(len(test_paths)):
    if test_paths[i].strip("/") == args.path.strip("/"):
        print(f"Average Final ({test_paths[i].split('.')[1]}): (same domain, {args.dropout:.3f}) overfitting AP = {average_best_train_result:.3f}, best valid AP = {average_best_valid_result_list[i]:.3f}, valid AP (training negatives) = {average_valid_result_train_neg_list[i]:.3f}, valid AP (test) = {average_valid_result_test_list[i]:.3f}")
    else:
        print(f"Average Final ({test_paths[i].split('.')[1]}): (cross domain, {args.dropout:.3f}) best valid AP = {average_best_valid_result_list[i]:.3f}, test AP = {average_test_result_list[i]:.3f}, test AP (training negatives) = {average_test_result_train_neg_list[i]:.3f}, test AP (test) = {average_test_result_test_list[i]:.3f}")
