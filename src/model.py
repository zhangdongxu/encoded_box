import numpy as np
import torch
import random
from collections import Counter
from torch import nn
import torch.nn.functional as F
from utils import log1mexp

random.seed(1234)

class vTE(nn.Module):
    # refer to "https://aclanthology.org/S18-1115/"
    def __init__(self, input_dim):

        super().__init__()
        self.layer = nn.Linear(input_dim, input_dim)
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def reset_parameters(self):
        self.layer.reset_parameters()

    def forward(self, idx1, idx2, emb1, emb2):
        return torch.log(torch.sigmoid(100 * self.cosine(self.layer(emb1), emb2)))


class CRIM(nn.Module):
    # refer to "https://aclanthology.org/S18-1115/"
    def __init__(self, input_dim, hidden_dim):

        super().__init__()
        self.init_tensor = torch.empty(hidden_dim, input_dim, input_dim)
        nn.init.normal_(self.init_tensor)
        self.mat = torch.nn.Parameter(self.init_tensor, requires_grad=True)
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def reset_parameters(self):
        self.mat.data = self.init_tensor
        self.linear.reset_parameters()

    def forward(self, idx1, idx2, emb1, emb2):
        hidden_feat = torch.matmul(emb1[None, :, :], self.mat) # (K, bz, dim)
        hid_feat = (hidden_feat * emb2[None, :, :]).sum(-1) # (K, bz)
        return torch.log(torch.sigmoid(self.linear(hid_feat.T).flatten()) + 1e-20)


class NeuralDot(nn.Module):
    # MLP + dot product
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout = 0.0, num_layers=3, num_shared_layers=0):
        # input -> shared_layers -> seperate layers -> in / out embeddings
        # if shared_layers == num_layers, then in and out embeddings become equal.

        if num_shared_layers > num_layers:
            raise Exception(f"shared_layers need to be less or equal to the num_layers")

        super().__init__()
        self.num_shared_layers = num_shared_layers
        self.num_layers = num_layers

        self.layers_shared = []
        for i in range(num_shared_layers):
            if i == 0 and i == num_layers - 1:
                layer = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers_shared.append(layer)

            if i < num_layers - 1:
                self.layers_shared.append(nn.ReLU())
                self.layers_shared.append(nn.Dropout(p=dropout))

        self.layers_shared = nn.ModuleList(self.layers_shared)

        self.layers1 = []
        self.layers2 = []
        for i in range(num_shared_layers, num_layers):
            if i == 0 and i == num_layers - 1:
                layer1 = nn.Linear(input_dim, output_dim)
                layer2 = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer1 = nn.Linear(input_dim, hidden_dim)
                layer2 = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer1 = nn.Linear(hidden_dim, output_dim)
                layer2 = nn.Linear(hidden_dim, output_dim)
            else:
                layer1 = nn.Linear(hidden_dim, hidden_dim)
                layer2 = nn.Linear(hidden_dim, hidden_dim)
            #torch.nn.init.eye_(layer2.weight)
            #torch.nn.init.zeros_(layer2.bias)

            self.layers1.append(layer1)
            self.layers2.append(layer2)
            if i < num_layers - 1:
                self.layers1.append(nn.ReLU())
                self.layers1.append(nn.Dropout(p=dropout))
                self.layers2.append(nn.ReLU())
                self.layers2.append(nn.Dropout(p=dropout))

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)

    def reset_parameters(self):
        for lin in self.layers_shared:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers1:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers2:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

    def forward(self, idx1, idx2, emb1, emb2):

        for layer in self.layers_shared:
            emb1 = layer(emb1)
            emb2 = layer(emb2)
        for layer in self.layers1:
            emb1 = layer(emb1)
        for layer in self.layers2:
            emb2 = layer(emb2)
        return torch.log(torch.sigmoid(torch.sum(emb1 * emb2, -1)) + 1e-20)
    

class NeuralCosine(nn.Module):
    # MLP + cosine similarity
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout = 0.0, num_layers=3, num_shared_layers=0):
        # input -> shared_layers -> seperate layers -> in / out embeddings
        # if shared_layers == num_layers, then in and out embeddings become equal.

        if num_shared_layers > num_layers:
            raise Exception(f"shared_layers need to be less or equal to the num_layers")

        super().__init__()
        self.num_shared_layers = num_shared_layers
        self.num_layers = num_layers
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.layers_shared = []
        for i in range(num_shared_layers):
            if i == 0 and i == num_layers - 1:
                layer = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers_shared.append(layer)

            if i < num_layers - 1:
                self.layers_shared.append(nn.ReLU())
                self.layers_shared.append(nn.Dropout(p=dropout))

        self.layers_shared = nn.ModuleList(self.layers_shared)

        self.layers1 = []
        self.layers2 = []
        for i in range(num_shared_layers, num_layers):
            if i == 0 and i == num_layers - 1:
                layer1 = nn.Linear(input_dim, output_dim)
                layer2 = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer1 = nn.Linear(input_dim, hidden_dim)
                layer2 = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer1 = nn.Linear(hidden_dim, output_dim)
                layer2 = nn.Linear(hidden_dim, output_dim)
            else:
                layer1 = nn.Linear(hidden_dim, hidden_dim)
                layer2 = nn.Linear(hidden_dim, hidden_dim)
            #torch.nn.init.eye_(layer2.weight)
            #torch.nn.init.zeros_(layer2.bias)

            self.layers1.append(layer1)
            self.layers2.append(layer2)
            if i < num_layers - 1:
                self.layers1.append(nn.ReLU())
                self.layers1.append(nn.Dropout(p=dropout))
                self.layers2.append(nn.ReLU())
                self.layers2.append(nn.Dropout(p=dropout))

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)

    def reset_parameters(self):
        for lin in self.layers_shared:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers1:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers2:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

    def forward(self, idx1, idx2, emb1, emb2):

        for layer in self.layers_shared:
            emb1 = layer(emb1)
            emb2 = layer(emb2)
        for layer in self.layers1:
            emb1 = layer(emb1)
        for layer in self.layers2:
            emb2 = layer(emb2)
        return torch.log(torch.sigmoid(100 * self.cosine(emb1, emb2)))
    

class NeuralComplex(nn.Module):
    # MLP + Complex ("Complex Embeddings for Simple Link Prediction")
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0, num_layers=3, num_shared_layers=0):
        # input -> shared_layers -> seperate layers -> Re / Im embeddings
        # if shared_layers == num_layers, then Re and Im embeddings become equal.
        if num_shared_layers > num_layers:
            raise Exception(f"shared_layers need to be less or equal to the num_layers, but now {num_shared_layers} > {num_layers}")

        super().__init__()
        self.asym = asym
        self.num_shared_layers = num_shared_layers
        self.num_layers = num_layers

        self.layers_shared = []
        for i in range(num_shared_layers):
            if i == 0 and i == num_layers - 1:
                layer = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            #torch.nn.init.eye_(layer.weight)
            #torch.nn.init.zeros_(layer.bias)
            self.layers_shared.append(layer)

            if i < num_layers - 1:
                self.layers_shared.append(nn.ReLU())
                self.layers_shared.append(nn.Dropout(p=dropout))

        self.layers_shared = nn.ModuleList(self.layers_shared)

        self.layers1 = []
        self.layers2 = []
        for i in range(num_shared_layers, num_layers):
            if i == 0 and i == num_layers - 1:
                layer1 = nn.Linear(input_dim, output_dim)
                layer2 = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer1 = nn.Linear(input_dim, hidden_dim)
                layer2 = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer1 = nn.Linear(hidden_dim, output_dim)
                layer2 = nn.Linear(hidden_dim, output_dim)
            else:
                layer1 = nn.Linear(hidden_dim, hidden_dim)
                layer2 = nn.Linear(hidden_dim, hidden_dim)

            self.layers1.append(layer1)
            self.layers2.append(layer2)
            if i < num_layers - 1:
                self.layers1.append(nn.ReLU())
                self.layers2.append(nn.ReLU())
                self.layers1.append(nn.Dropout(p=dropout))
                self.layers2.append(nn.Dropout(p=dropout))

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)

    def reset_parameters(self):
        for lin in self.layers_shared:
            lin.reset_parameters()
        for lin in self.layers1:
            lin.reset_parameters()
        for lin in self.layers2:
            lin.reset_parameters()

    def forward(self, idx1, idx2, emb1, emb2):

        for layer in self.layers_shared:
            emb1 = layer(emb1)
            emb2 = layer(emb2)
        emb11, emb12 = emb1, emb1
        emb21, emb22 = emb2, emb2
        for layer in self.layers1:
            emb11 = layer(emb11)
            emb21 = layer(emb21)
        for layer in self.layers2:
            emb12 = layer(emb12)
            emb22 = layer(emb22)
        score = torch.sum(emb11 * emb21, -1) + \
                torch.sum(emb12 * emb22, -1) + \
                torch.sum(emb11 * emb22, -1) - \
                torch.sum(emb12 * emb21, -1)
        return torch.log(torch.sigmoid(score) + 1e-20)


class NeuralPOE(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0, num_layers=3):
        # MLP + POE

        super().__init__()

        self.layers = []
        for i in range(num_layers):
            if i == 0 and i == num_layers - 1:
                layer = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers.append(layer)

            if i < num_layers - 1:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout))
        self.layers = nn.ModuleList(self.layers)

    def reset_parameters(self):
        for lin in self.layers:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

    def log_volume(self, z):
        return -z.sum(-1)

    def intersection(self, z1, z2):
        return torch.max(z1, z2)

    def forward(self, idx1, idx2, emb1, emb2):
        for layer in self.layers:
            emb1 = layer(emb1)
            emb2 = layer(emb2)

        meet_z = self.intersection(emb1, emb2)

        log_overlap_volume = self.log_volume(meet_z)
        log_rhs_volume = self.log_volume(emb2)

        logp = log_overlap_volume - log_rhs_volume
        return logp


class NeuralBox(nn.Module):
    # MLP + Gumbel Box
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0, num_layers=3, num_shared_layers=0, volume_temp=1.0, intersection_temp=0.01):
        # input -> shared_layers -> seperate layers -> Box_z / Box_Z embeddings
        # if shared_layers == num_layers, then Box_z and Box_Z embeddings become equal.
        if num_shared_layers > num_layers:
            raise Exception(f"shared_layers need to be less or equal to the num_layers")

        super().__init__()
        self.num_shared_layers = num_shared_layers
        self.num_layers = num_layers
        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.layers_shared = []
        for i in range(num_shared_layers):
            if i == 0 and i == num_layers - 1:
                layer = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers_shared.append(layer)

            if i < num_layers - 1:
                self.layers_shared.append(nn.ReLU())
                self.layers_shared.append(nn.Dropout(p=dropout))
        self.layers_shared = nn.ModuleList(self.layers_shared)

        self.layers1 = []
        self.layers2 = []
        for i in range(num_shared_layers, num_layers):
            if i == 0 and i == num_layers - 1:
                layer1 = nn.Linear(input_dim, output_dim)
                layer2 = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer1 = nn.Linear(input_dim, hidden_dim)
                layer2 = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer1 = nn.Linear(hidden_dim, output_dim)
                layer2 = nn.Linear(hidden_dim, output_dim)
            else:
                layer1 = nn.Linear(hidden_dim, hidden_dim)
                layer2 = nn.Linear(hidden_dim, hidden_dim)

            self.layers1.append(layer1)
            self.layers2.append(layer2)
            if i < num_layers - 1:
                self.layers1.append(nn.ReLU())
                self.layers2.append(nn.ReLU())
                self.layers1.append(nn.Dropout(p=dropout))
                self.layers2.append(nn.Dropout(p=dropout))

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)

    def reset_parameters(self):
        for lin in self.layers_shared:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers1:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers2:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

    def log_volume(self, z, Z):
        log_vol = torch.sum(
            torch.log(self.volume_temp * self.softplus((Z - z - self.softplus_const) / self.volume_temp) + 1e-20), dim=-1,
        )
        return log_vol

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -
                    e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idx1, idx2, emb1, emb2):
        for layer in self.layers_shared:
            emb1 = layer(emb1)
            emb2 = layer(emb2)
        emb1_z, emb1_Z = emb1, emb1
        emb2_z, emb2_Z = emb2, emb2
        for layer in self.layers1:
            emb1_z = layer(emb1_z)
            emb2_z = layer(emb2_z)
        for layer in self.layers2:
            emb1_Z = layer(emb1_Z)
            emb2_Z = layer(emb2_Z)
        cen1 = emb1_z
        sl1 = self.softplus(emb1_Z) / 2
        min1 = cen1 - sl1
        max1 = cen1 + sl1
        cen2 = emb2_z
        sl2 = self.softplus(emb2_Z) / 2
        min2 = cen2 - sl2
        max2 = cen2 + sl2

        meet_min, meet_max = self.gumbel_intersection(
            min1, max1, min2, max2)

        log_overlap_volume = self.log_volume(
            meet_min, meet_max)
        log_rhs_volume = self.log_volume(min2, max2)

        logp = log_overlap_volume - log_rhs_volume
        return logp


class NeuralVBCBox(nn.Module):
    # MLP + Binary Code Box Embeddings
    def __init__(self, input_dim, output_dim, hidden_dim=128, dropout=0.0, num_layers=3, num_shared_layers=0, volume_temp=1.0, intersection_temp=0.01):
        # input -> shared_layers -> seperate layers -> Box_z / Box_Z embeddings
        # if shared_layers == num_layers, then Box_z and Box_Z embeddings become equal.
        if num_shared_layers > num_layers:
            raise Exception(f"shared_layers need to be less or equal to the num_layers")

        super().__init__()
        self.num_shared_layers = num_shared_layers
        self.num_layers = num_layers
        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.layers_shared = []
        for i in range(num_shared_layers):
            if i == 0 and i == num_layers - 1:
                layer = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            self.layers_shared.append(layer)

            if i < num_layers - 1:
                self.layers_shared.append(nn.ReLU())
                self.layers_shared.append(nn.Dropout(p=dropout))
        self.layers_shared = nn.ModuleList(self.layers_shared)

        self.layers1 = []
        self.layers2 = []
        self.layers3 = []
        for i in range(num_shared_layers, num_layers):
            if i == 0 and i == num_layers - 1:
                layer1 = nn.Linear(input_dim, output_dim)
                layer2 = nn.Linear(input_dim, output_dim)
                layer3 = nn.Linear(input_dim, output_dim)
            elif i == 0 and i < num_layers - 1:
                layer1 = nn.Linear(input_dim, hidden_dim)
                layer2 = nn.Linear(input_dim, hidden_dim)
                layer3 = nn.Linear(input_dim, hidden_dim)
            elif i == num_layers - 1:
                layer1 = nn.Linear(hidden_dim, output_dim)
                layer2 = nn.Linear(hidden_dim, output_dim)
                layer3 = nn.Linear(hidden_dim, output_dim)
            else:
                layer1 = nn.Linear(hidden_dim, hidden_dim)
                layer2 = nn.Linear(hidden_dim, hidden_dim)
                layer3 = nn.Linear(hidden_dim, hidden_dim)

            self.layers1.append(layer1)
            self.layers2.append(layer2)
            self.layers3.append(layer3)
            if i < num_layers - 1:
                self.layers1.append(nn.ReLU())
                self.layers2.append(nn.ReLU())
                self.layers3.append(nn.ReLU())
                self.layers1.append(nn.Dropout(p=dropout))
                self.layers2.append(nn.Dropout(p=dropout))
                self.layers3.append(nn.Dropout(p=dropout))

        self.layers1 = nn.ModuleList(self.layers1)
        self.layers2 = nn.ModuleList(self.layers2)
        self.layers3 = nn.ModuleList(self.layers3)

    def reset_parameters(self):
        for lin in self.layers_shared:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers1:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers2:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.layers3:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

    def log_volume(self, z, Z, c):
        log_vol = torch.sum(
            torch.log(self.volume_temp * self.softplus((Z - z - self.softplus_const) / self.volume_temp) + 1e-20) * c, dim=-1,
        )
        return log_vol

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -
                    e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idx1, idx2, emb1, emb2):
        bin1, bin2 = emb1, emb2
        for layer in self.layers_shared:
            emb1 = layer(emb1)
            emb2 = layer(emb2)
            bin1 = layer(bin1)
            bin2 = layer(bin2)
        emb1_z, emb1_Z = emb1, emb1
        emb2_z, emb2_Z = emb2, emb2
        for layer in self.layers1:
            emb1_z = layer(emb1_z)
            emb2_z = layer(emb2_z)
        for layer in self.layers2:
            emb1_Z = layer(emb1_Z)
            emb2_Z = layer(emb2_Z)
        for layer in self.layers3:
            bin1 = layer(bin1)
            bin2 = layer(bin2)
        cen1 = emb1_z
        sl1 = self.softplus(emb1_Z) / 2
        min1 = cen1 - sl1
        max1 = cen1 + sl1
        cen2 = emb2_z
        sl2 = self.softplus(emb2_Z) / 2
        min2 = cen2 - sl2
        max2 = cen2 + sl2
        bin_vec = self.sigmoid(bin1) * self.sigmoid(bin2)

        meet_min, meet_max = self.gumbel_intersection(
            min1, max1, min2, max2)

        log_overlap_volume = self.log_volume(
            meet_min, meet_max, bin_vec)
        log_rhs_volume = self.log_volume(min2, max2, bin_vec)

        logp = log_overlap_volume - log_rhs_volume
        return logp


class Dot(nn.Module):
    def __init__(self, dim, N):

        super().__init__()
        self.embs1 = torch.nn.Embedding(N, dim)
        self.embs2 = torch.nn.Embedding(N, dim)

    def reset_parameters(self):
        self.embs1.reset_parameters()
        self.embs2.reset_parameters()


    def forward(self, idx1, idx2, emb1, emb2):
        emb1 = self.embs1(idx1)
        emb2 = self.embs2(idx2)
        return torch.log(torch.sigmoid(torch.sum(emb1 * emb2, -1)) + 1e-20)
    
class Cosine(nn.Module):
    def __init__(self, dim, N):

        super().__init__()
        self.embs1 = torch.nn.Embedding(N, dim)
        self.embs2 = torch.nn.Embedding(N, dim)
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    def reset_parameters(self):
        self.embs1.reset_parameters()
        self.embs2.reset_parameters()


    def forward(self, idx1, idx2, emb1, emb2):
        emb1 = self.embs1(idx1)
        emb2 = self.embs2(idx2)
        return torch.log(torch.sigmoid(100 * self.cosine(emb1, emb2)))


class Box(nn.Module):
    def __init__(self, dim, N, volume_temp=1.0, intersection_temp=0.01):
        super().__init__()
        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.embs1 = torch.nn.Embedding(N, dim)
        self.embs2 = torch.nn.Embedding(N, dim)


    def reset_parameters(self):
        self.embs1.reset_parameters()
        self.embs2.reset_parameters()

    def log_volume(self, z, Z):
        log_vol = torch.sum(
            torch.log(self.volume_temp * self.softplus((Z - z - self.softplus_const) / self.volume_temp) + 1e-20), dim=-1,
        )
        return log_vol

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -
                    e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idx1, idx2, emb1, emb2):
        c1 = self.embs1(idx1)
        w1 = self.softplus(self.embs2(idx1)) / 2
        min1 = c1 - w1
        max1 = c1 + w1
        c2 = self.embs1(idx2)
        w2 = self.softplus(self.embs2(idx2)) / 2
        min2 = c2 - w2
        max2 = c2 + w2

        meet_min, meet_max = self.gumbel_intersection(
            min1, max1, min2, max2)

        log_overlap_volume = self.log_volume(
            meet_min, meet_max)
        log_rhs_volume = self.log_volume(min2, max2)

        logp = log_overlap_volume - log_rhs_volume
        return logp


class VBCBox(nn.Module):
    def __init__(self, dim, N, volume_temp=1.0, intersection_temp=0.01):
        super().__init__()
        self.volume_temp = volume_temp
        self.intersection_temp = intersection_temp
        self.softplus_const = 2 * self.intersection_temp * 0.57721566490153286060
        self.softplus = torch.nn.Softplus()
        self.sigmoid = torch.nn.Sigmoid()

        self.embs1 = torch.nn.Embedding(N, dim)
        self.embs2 = torch.nn.Embedding(N, dim)
        self.bins = torch.nn.Embedding(N, dim)


    def reset_parameters(self):
        self.embs1.reset_parameters()
        self.embs2.reset_parameters()
        self.bins.reset_parameters()

    def log_volume(self, z, Z, c):
        log_vol = torch.sum(
            torch.log(self.volume_temp * self.softplus((Z - z - self.softplus_const) / self.volume_temp) + 1e-20) * c, dim=-1,
        )
        return log_vol

    def gumbel_intersection(self, e1_min, e1_max, e2_min, e2_max):
        meet_min = self.intersection_temp * torch.logsumexp(
            torch.stack(
                [e1_min / self.intersection_temp, e2_min / self.intersection_temp]
            ),
            0,
        )
        meet_max = -self.intersection_temp * torch.logsumexp(
            torch.stack(
                [-e1_max / self.intersection_temp, -
                    e2_max / self.intersection_temp]
            ),
            0,
        )
        meet_min = torch.max(meet_min, torch.max(e1_min, e2_min))
        meet_max = torch.min(meet_max, torch.min(e1_max, e2_max))
        return meet_min, meet_max

    def forward(self, idx1, idx2, emb1, emb2):
        c1 = self.embs1(idx1)
        w1 = self.softplus(self.embs2(idx1)) / 2
        min1 = c1 - w1
        max1 = c1 + w1
        c2 = self.embs1(idx2)
        w2 = self.softplus(self.embs2(idx2)) / 2
        min2 = c2 - w2
        max2 = c2 + w2
        bin1 = self.bins(idx1)
        bin2 = self.bins(idx2)
        bin_vec = self.sigmoid(bin1) * self.sigmoid(bin2)

        meet_min, meet_max = self.gumbel_intersection(
            min1, max1, min2, max2)

        log_overlap_volume = self.log_volume(
            meet_min, meet_max, bin_vec)
        log_rhs_volume = self.log_volume(min2, max2, bin_vec)

        logp = log_overlap_volume - log_rhs_volume
        return logp
