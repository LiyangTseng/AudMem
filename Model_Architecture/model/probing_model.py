import numpy as np
import torch
import torch.nn as nn

class Highway(nn.Module):
    ''' ref: what does the network layer hear '''
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H = self.relu(self.H(x))
        T = self.sigmoid(self.T(x))
        y = H * T + x * (1.0 - T)
        return y

class Probing_Model(nn.Module):
    """ use to reconstruct audio """
    def __init__(self, model_config, mode="train"):
        super().__init__()
        if mode in ["train", "test"]:
            self.mode = mode
        else:
            raise Exception("Invalid model mode")
        self.downsampling_factor = model_config["downsampling_factor"]
        self.output_size = model_config["output_size"]

        self.Batch_Norm = nn.BatchNorm1d(model_config["input_size"])
        hidden_dim = model_config["output_size"]*self.downsampling_factor
        self.Projection = nn.Linear(model_config["input_size"], hidden_dim)
        # self.Highway_Network = Highway(size=model_config["output_size"]*self.downsampling_factor, num_layers=model_config["layer_num"])
        self.Highway_Network = nn.Sequential(
            *((model_config["layer_num"] - 1) * [Highway(hidden_dim, hidden_dim)]),   
        )
        self.Relu_1 = nn.ReLU()
        # self.Linear = nn.Linear(model_config["input_size"], model_config["output_size"]*self.downsampling_factor)
        self.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.Relu_2 = nn.ReLU()
        self.Loss_Function = nn.L1Loss()

    def forward(self, *args):
        if self.mode == "train":
            (features, labels) = args
            # features = self.Batch_Norm(features)
            out = self.Projection(features)
            out = self.Highway_Network(out)
            # out = self.Relu_1(out)
            out = self.Linear(out)
            # reshape (upsampling)
            predictions = out.reshape(-1, out.size(-1) // self.downsampling_factor, self.downsampling_factor)
            # predictions = self.Relu_2(out)
            loss = self.Loss_Function(predictions, labels)
            return predictions, loss
        else:
            (features, ) = args
            out = self.Projection(features)
            out = self.Highway_Network(out)
            # out = self.Relu_1(out)
            out = self.Linear(out)
            # reshape (upsampling)
            predictions = out.reshape(-1, out.size(-1) // self.downsampling_factor, self.downsampling_factor)# reshape (upsampling)
            return predictions

