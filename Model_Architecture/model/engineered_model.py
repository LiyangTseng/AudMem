import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, sequential_input_size, non_sequential_input_size, hidden_size):
        super(MLP, self).__init__()
        print("Using MLP")
        self.input_size = sequential_input_size + non_sequential_input_size
        self.Linear_1 = nn.Linear(self.input_size, hidden_size)
        self.Relu = nn.ReLU()
        self.Linear_2 = nn.Linear(hidden_size, 1)
        self.Sigmoid = nn.Sigmoid()

        self.Loss_Function = nn.BCELoss()

    def forward(self, sequential_features, non_sequential_features, labels):
        features = torch.cat((sequential_features, non_sequential_features), 1)
        x = self.Linear_1(features)
        x = self.Relu(x)
        x = self.Linear_2(x)
        outputs = self.Sigmoid(x)
        predictions = torch.max(outputs, 1)[1] # return the indices for the max through axis 1 
        correct_count = (predictions == labels).sum()
        loss = self.Loss_Function(outputs, labels.reshape(-1,1))
        batch_size = labels.size(0)

        return batch_size, correct_count, loss

class LSTM(nn.Module):
    def __init__(self, sequential_input_size, non_sequential_input_size, hidden_size, num_layers, device, bidirectional=True):
        super().__init__()
        print("Using LSTM")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.bidirectional = bidirectional
        self.LSTM = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_size = sequential_input_size if i == 0 else intermediate_hidden_size*(1+int(self.bidirectional))
            intermediate_hidden_size = self.hidden_size//(2**i)
            self.LSTM.append(nn.LSTM(input_size=input_size, hidden_size=intermediate_hidden_size, batch_first=True, bidirectional=True))

        # input shape: (batch_size, seq, input_size)
        self.Linear = nn.Linear(self.hidden_size//8*(1+int(self.bidirectional))+non_sequential_input_size, 1)

        self.Sigmoid = nn.Sigmoid()
        self.Loss_Function = nn.BCELoss()

    def forward(self, sequential_features, non_sequential_features, labels):

        for idx, layer in enumerate(self.LSTM):
            inputs = sequential_features if idx == 0 else hidden_states
            h0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size//(2**idx), dtype=torch.double, device=self.device)
            c0 = torch.zeros(1+int(self.bidirectional), inputs.size(0), self.hidden_size//(2**idx), dtype=torch.double, device=self.device)
            # in shape: (batch_size, seq_len, input_size)
            hidden_states, _ = layer(inputs, (h0, c0))
            # out shape: (batch_size, seq_length, hidden_size*bidirectional)
    
        # only use the last timestep for linear input
        out = hidden_states[:, -1, :]
        out = torch.cat((out, non_sequential_features), 1)
        out = self.Linear(out)

        outputs = self.Sigmoid(out)
        predictions = torch.max(outputs, 1)[1] # return the indices for the max through axis 1 
        correct_count = (predictions == labels).sum()
        loss = self.Loss_Function(outputs, labels.reshape(-1,1))
        batch_size = labels.size(0)

        return batch_size, correct_count, loss
    
if __name__ == "__main__":
    # model = MLP()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sequential_input_size = 408
    non_sequential_input_size = 2
    model = LSTM(sequential_input_size, non_sequential_input_size, hidden_size=640, num_layers=4, device=device, bidirectional=True).to(device).double()
