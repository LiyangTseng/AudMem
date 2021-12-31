import torch
import torch.nn as nn
import torch.nn.functional as F

''' train features to approximate memorability score (regression) '''

class H_MLP(nn.Module):
    '''
        MLP model for handcrafted features
    '''
    def __init__(self, model_config, device, mode="train"):
        if mode in ["train", "test"]:
            self.mode = mode
        else:
            raise Exception("Invalid model mode")

        super(H_MLP, self).__init__()
        print("Using MLP")
        self.device = device
        self.input_size = model_config["sequential_input_size"] + model_config["non_sequential_input_size"]
        self.Linear_1 = nn.Linear(self.input_size, model_config["hidden_size"])
        self.Relu = nn.ReLU()
        self.BatchNorm = nn.BatchNorm1d(model_config["hidden_size"])
        self.Linear_2 = nn.Linear(model_config["hidden_size"], 1)
        self.Sigmoid = nn.Sigmoid()

        self.Loss_Function = nn.MSELoss()

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| H_MLP: average pooling on time axis of sequential features, concate all features to MLP')
        # TODO: add one regarding attention
        # if self.encoder.vgg:
        #     msg.append('           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
        # if self.enable_ctc:
        #     msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(self.ctc_weight))
        # if self.enable_att:
        #     msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(self.attention.mode,1-self.ctc_weight))
        return msg

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
            return predictions, labels
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

class H_LSTM(nn.Module):
    '''
        LSTM model for handcrafted features
    '''

    def __init__(self, model_config, device):
        super().__init__()

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

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| H_LSTM: apply LSTM to sequential features, involve non sequential features in output layer ')
        # TODO: add one regarding attention
        # if self.encoder.vgg:
        #     msg.append('           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
        # if self.enable_ctc:
        #     msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(self.ctc_weight))
        # if self.enable_att:
        #     msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(self.attention.mode,1-self.ctc_weight))
        return msg

    def forward(self, sequential_features, non_sequential_features):

        for idx, layer in enumerate(self.LSTM):
            inputs = sequential_features if idx == 0 else hidden_states
            h0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size, device=self.device)
            c0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size, device=self.device)
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


if __name__ == "__main__":
    # model = MLP()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sequential_input_size = 408
    non_sequential_input_size = 2
    model = H_LSTM(sequential_input_size, non_sequential_input_size, hidden_size=640, num_layers=4, device=device, bidirectional=True).to(device)
