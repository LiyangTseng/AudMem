import torch
import torch.nn as nn

class Highway(nn.Module):
    ''' ref: https://github.com/kefirski/pytorch_Highway '''
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
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x

class Probing_Model(nn.Module):
    """ use to reconstruct audio """
    def __init__(self, model_config, mode="train"):
        super().__init__()
        if mode in ["train", "test"]:
            self.mode = mode
        else:
            raise Exception("Invalid model mode")

        self.Batch_Norm = nn.BatchNorm1d(640)
        self.Highway_Network = Highway(size=model_config["input_size"], num_layers=model_config["layer_num"], f=torch.relu)
        self.Relu_1 = nn.ReLU()
        self.Linear = nn.Linear(model_config["input_size"], model_config["output_size"])
        self.Relu_2 = nn.ReLU()
        self.Loss_Function = nn.L1Loss()

    # def forward(self, features, labels):
    def forward(self, *args):
        if self.mode == "train":
            (features, labels) = args
            features = self.Batch_Norm(features)
            out = self.Highway_Network(features)
            out = self.Relu_1(out)
            out = self.Linear(out)
            predictions = self.Relu_2(out)
            loss = self.Loss_Function(predictions, labels)
            return predictions, loss
        else:
            (features, ) = args
            features = self.Batch_Norm(features)
            out = self.Highway_Network(features)
            out = self.Relu_1(out)
            out = self.Linear(out)
            predictions = self.Relu_2(out)
            return predictions