import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_NUM = 410

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.Linear_1 = nn.Linear(FEATURE_NUM, 640)
        self.Relu = nn.ReLU()
        self.Linear_2 = nn.Linear(640, 1)
        self.Sigmoid = nn.Sigmoid()

        self.Loss_Function = nn.BCELoss()

    def forward(self, features, labels):
        x = self.Linear_1(features)
        x = self.Relu(x)
        x = self.Linear_2(x)
        outputs = self.Sigmoid(x)
        predictions = torch.max(outputs, 1)[1] # return the indices for the max through axis 1 
        correct_count = (predictions == labels).sum()
        loss = self.Loss_Function(outputs, labels.reshape(-1,1))
        batch_size = labels.size(0)

        return batch_size, correct_count, loss
class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x

class Probing_Model():
    """ use to reconstruct audio """
    def __init__(self):
        super().__init__()
        self.Linear_Projection = nn.Linear()

    def forwar(self, hidden_states):
        pass

if __name__ == "__main__":
    model = MLP()