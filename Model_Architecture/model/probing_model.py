import numpy as np
import torch
import torch.nn as nn

def init_linear(input_linear):
    ''' ref: https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/blob/master/model/utils.py '''
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()

class Highway(nn.Module):
    ''' ref: https://github.com/LiyuanLucasLiu/LM-LSTM-CRF/blob/master/model/highway.py '''
    def __init__(self, size, num_layers = 1, dropout_ratio = 0.5):
        super(Highway, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.trans = nn.ModuleList()
        self.gate = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_ratio)

        for i in range(num_layers):
            tmptrans = nn.Linear(size, size)
            tmpgate = nn.Linear(size, size)
            self.trans.append(tmptrans)
            self.gate.append(tmpgate)

    def rand_init(self):
        """
        random initialization
        """
        for i in range(self.num_layers):
            init_linear(self.trans[i])
            init_linear(self.gate[i])

    def forward(self, x):
        """
        update statics for f1 score
        args: 
            x (ins_num, hidden_dim): input tensor
        return:
            output tensor (ins_num, hidden_dim)
        """
        
        
        g = torch.sigmoid(self.gate[0](x))
        h = nn.functional.relu(self.trans[0](x))
        x = g * h + (1 - g) * x

        for i in range(1, self.num_layers):
            x = self.dropout(x)
            g = torch.sigmoid(self.gate[i](x))
            h = nn.functional.relu(self.trans[i](x))
            x = g * h + (1 - g) * x

        return x
class Probing_Model(nn.Module):
    """ use to reconstruct audio """
    def __init__(self, model_config, downsampling_factor=1, mode="train"):
        super().__init__()
        if mode in ["train", "test"]:
            self.mode = mode
        else:
            raise Exception("Invalid model mode")
        self.downsampling_factor = downsampling_factor
        self.output_size = model_config["output_size"]

        self.Batch_Norm = nn.BatchNorm1d(model_config["input_size"])
        self.Highway_Network = Highway(size=model_config["input_size"], num_layers=model_config["layer_num"])
        self.Relu_1 = nn.ReLU()
        self.Linear = nn.Linear(model_config["input_size"], model_config["output_size"]*self.downsampling_factor)
        self.Relu_2 = nn.ReLU()
        self.Loss_Function = nn.L1Loss()

    def forward(self, *args):
        if self.mode == "train":
            (features, labels) = args
            features = self.Batch_Norm(features)
            out = self.Highway_Network(features)
            out = self.Relu_1(out)
            out = self.Linear(out)
            # TODO: check with original author
            # reshape (upsampling)
            out = out.view(out.size(0), -1, self.output_size)
            predictions = self.Relu_2(out)
            loss = self.Loss_Function(predictions, labels)
            return predictions, loss
        else:
            (features, ) = args
            features = self.Batch_Norm(features)
            out = self.Highway_Network(features)
            out = self.Relu_1(out)
            out = self.Linear(out)
            # reshape (upsampling)
            out = out.view(out.size(0), -1, self.output_size)
            predictions = self.Relu_2(out)
            return predictions

