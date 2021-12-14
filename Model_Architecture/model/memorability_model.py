import torch
import torch.nn as nn
import torch.nn.functional as F

''' train features to approximate memorability score (regression) '''

class Regression_MLP(nn.Module):
    def __init__(self, model_config, device, mode="train"):
        if mode in ["train", "test"]:
            self.mode = mode
        else:
            raise Exception("Invalid model mode")

        super(Regression_MLP, self).__init__()
        print("Using MLP")
        self.device = device
        self.input_size = model_config["sequential_input_size"] + model_config["non_sequential_input_size"]
        self.Linear_1 = nn.Linear(self.input_size, model_config["hidden_size"])
        self.Relu = nn.ReLU()
        self.BatchNorm = nn.BatchNorm1d(model_config["hidden_size"])
        self.Linear_2 = nn.Linear(model_config["hidden_size"], 1)
        self.Sigmoid = nn.Sigmoid()

        self.Loss_Function = nn.MSELoss()

    def forward(self, data):
        if self.mode == "train":
            sequential_features, non_sequential_features, labels = data
            sequential_features, non_sequential_features, labels = sequential_features.to(self.device), non_sequential_features.to(self.device), labels.to(self.device)
            
            features = torch.cat((sequential_features, non_sequential_features), 1)
            x = self.Linear_1(features)
            x = self.Relu(x)
            # x = self.BatchNorm(x)
            x = self.Linear_2(x)
            predictions = self.Sigmoid(x)
            loss = self.Loss_Function(predictions, labels.reshape(-1,1))

            return predictions, loss
        else:
            sequential_features, non_sequential_features = data
            sequential_features, non_sequential_features = sequential_features.to(self.device), non_sequential_features.to(self.device)
            
            features = torch.cat((sequential_features, non_sequential_features), 1)
            x = self.Linear_1(features)
            x = self.Relu(x)
            # x = self.BatchNorm(x)
            x = self.Linear_2(x)
            predictions = self.Sigmoid(x)

            return predictions

class Regression_LSTM(nn.Module):
    def __init__(self, model_config, device, mode="train"):
        super().__init__()
        if mode in ["train", "test"]:
            self.mode = mode
        else:
            raise Exception("Invalid model mode")

        print("Using LSTM")
        self.hidden_size = model_config["hidden_size"]
        self.num_layers = model_config["layer_num"]
        self.device = device
        self.bidirectional = model_config["bidirectional"]
        
        self.Batch_Norm_1 = nn.BatchNorm1d(model_config["seq_len"])
        self.LSTM = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_size = model_config["sequential_input_size"] if i == 0 else intermediate_hidden_size*(1+int(self.bidirectional))
            intermediate_hidden_size = self.hidden_size
            self.LSTM.append(nn.LSTM(input_size=input_size, hidden_size=intermediate_hidden_size, batch_first=True, bidirectional=True))

        self.Batch_Norm_2 = nn.BatchNorm1d(2*model_config["hidden_size"]+model_config["non_sequential_input_size"])
        # input shape: (batch_size, seq, input_size)
        self.Linear = nn.Linear(self.hidden_size*(1+int(self.bidirectional))+model_config["non_sequential_input_size"], 1)

        self.Sigmoid = nn.Sigmoid()
        self.Loss_Function = nn.MSELoss()

    def forward(self, data):
        if self.mode == "train":
            sequential_features, non_sequential_features, labels = data
            sequential_features, non_sequential_features, labels = sequential_features.to(self.device), non_sequential_features.to(self.device), labels.to(self.device)

            # sequential_features = self.Batch_Norm_1(sequential_features)
            for idx, layer in enumerate(self.LSTM):
                inputs = sequential_features if idx == 0 else hidden_states
                h0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size, dtype=torch.double, device=self.device)
                c0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size, dtype=torch.double, device=self.device)
                # in shape: (batch_size, seq_len, input_size)
                hidden_states, _ = layer(inputs, (h0, c0))
                # out shape: (batch_size, seq_length, hidden_size*bidirectional)
        
            # only use the last timestep for linear input
            out = hidden_states[:, -1, :]
            out = torch.cat((out, non_sequential_features), 1)
            out = self.Batch_Norm_2(out)
            out = self.Linear(out)

            predictions = self.Sigmoid(out)            
            loss = self.Loss_Function(predictions, labels.reshape(-1,1))

            return predictions, loss
        else:
            sequential_features, non_sequential_features = data
            sequential_features, non_sequential_features = sequential_features.to(self.device), non_sequential_features.to(self.device)

            sequential_features = self.Batch_Norm_1(sequential_features)
            for idx, layer in enumerate(self.LSTM):
                inputs = sequential_features if idx == 0 else hidden_states
                h0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size, dtype=torch.double, device=self.device)
                c0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size, dtype=torch.double, device=self.device)
                # in shape: (batch_size, seq_len, input_size)
                hidden_states, _ = layer(inputs, (h0, c0))
                # out shape: (batch_size, seq_length, hidden_size*bidirectional)
        
            # only use the last timestep for linear input
            out = hidden_states[:, -1, :]
            out = torch.cat((out, non_sequential_features), 1)
            out = self.Batch_Norm_2(out)
            out = self.Linear(out)

            predictions = self.Sigmoid(out)
            
            return predictions


''' train featrue pairs to approximate memorability ranking '''

class RankNet(nn.Module):
    ''' ref: https://www.shuzhiduo.com/A/qVdelZZp5P/ '''
    def __init__(self, model_config, device, backbone="LSTM", mode="train"):
        print("Using RankNet")
        super().__init__()
        if mode in ["train", "test"]:
            self.mode = mode
        else:
            raise Exception("Invalid model mode")
        self.device = device
        if backbone == "LSTM":
            self.model = Regression_LSTM(model_config, device, mode="test")
        elif backbone == "MLP":
            self.model = Regression_MLP(model_config, device, mode="test")
        else:
            raise Exception ("Invalid ranking backbone")

        self.sigmoid = nn.Sigmoid()
        self.Loss_Function = nn.BCELoss()

    def forward(self, data):
        if self.mode == "train":
            sequential_features_1, non_sequential_features_1, labels_1,\
                sequential_features_2, non_sequential_features_2, labels_2 = data
            result_1 = self.model((sequential_features_1, non_sequential_features_1)) # predicted score of input_1
            result_2 = self.model((sequential_features_2, non_sequential_features_2)) # predicted score of input_2
            predictions = self.sigmoid(result_1 - result_2)
            labels = torch.unsqueeze((labels_1>labels_2).double(), 1).to(self.device)
            # true/false of prediction
            results = torch.logical_and( labels, (result_1>result_2).to(self.device))
            loss = self.Loss_Function(predictions, labels)
            return predictions, results, loss
        else:
            sequential_features_1, non_sequential_features_1,\
                sequential_features_2, non_sequential_features_2  = data
            if len(sequential_features_1.shape) == 1:
                # not forward by batch yet
                sequential_features_1, non_sequential_features_1 = torch.unsqueeze(sequential_features_1, 0), torch.unsqueeze(non_sequential_features_1, 0)
                sequential_features_2, non_sequential_features_2 = torch.unsqueeze(sequential_features_2, 0), torch.unsqueeze(non_sequential_features_2, 0)
            result_1 = self.model((sequential_features_1, non_sequential_features_1)) # predicted score of input_1
            result_2 = self.model((sequential_features_2, non_sequential_features_2)) # predicted score of input_2
            return result_1 > result_2

if __name__ == "__main__":
    # model = MLP()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sequential_input_size = 408
    non_sequential_input_size = 2
    model = Regression_LSTM(sequential_input_size, non_sequential_input_size, hidden_size=640, num_layers=4, device=device, bidirectional=True).to(device).double()
