import numpy as np
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
            # loss = self.Loss_Function(predictions, labels.reshape(-1,1))

            # return predictions, loss
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

        self.att_weight = nn.Parameter(torch.randn(1, 2*self.hidden_size, 1))

    def attention_layer(self, h):
        '''
            temporal attetion mechanism
            from: https://github.com/onehaitao/Att-BLSTM-relation-extraction/blob/master/model.py 
        '''
        att_weight = self.att_weight.expand(h.shape[0], -1, -1)  # B*H*1
        att_score = torch.bmm(nn.Tanh()(h), att_weight)  # B*L*H  *  B*H*1 -> B*L*1

        att_weight = F.softmax(att_score, dim=1)  # B*L*1

        reps = torch.bmm(h.transpose(1, 2), att_weight).squeeze(dim=-1)  # B*H*L *  B*L*1 -> B*H*1 -> B*H
        reps = nn.Tanh()(reps)  # B*H
        return reps, att_weight

    def forward(self, data):
        if self.mode == "train":
            sequential_features, non_sequential_features, labels = data
            sequential_features, non_sequential_features, labels = sequential_features.to(self.device), non_sequential_features.to(self.device), labels.to(self.device)

            # sequential_features = self.Batch_Norm_1(sequential_features)
            for idx, layer in enumerate(self.LSTM):
                inputs = sequential_features if idx == 0 else output
                # in shape: (batch_size, seq_len, input_size)
                output, (_, _) = layer(inputs)
                # out shape: (batch_size, seq_length, hidden_size*bidirectional)
        
            att_out, att_weight = self.attention_layer(output)
            
            # TODO: add attention mechanism to decode all hidden states
            # only use the last timestep for linear input
            out = att_out[:, -1, :]
            out = torch.cat((out, non_sequential_features), 1)
            out = self.Batch_Norm_2(out)
            out = self.Linear(out)

            predictions = self.Sigmoid(out)            
            # loss = self.Loss_Function(predictions, labels.reshape(-1,1))

            # return predictions, loss
            return predictions, labels
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

class BaseAttention(nn.Module):
    ''' Base module for attentions '''
    def __init__(self, temperature, num_head):
        super().__init__()
        self.temperature = temperature
        self.num_head = num_head
        self.softmax = nn.Softmax(dim=-1)
        self.reset_mem()

    def reset_mem(self):
        # Reset mask
        self.mask = None
        self.k_len = None

    def set_mem(self):
        pass

    def compute_mask(self,k,k_len):
        # Make the mask for padded states
        self.k_len = k_len
        bs,ts,_ = k.shape
        self.mask = np.zeros((bs,self.num_head,ts))
        for idx,sl in enumerate(k_len):
            self.mask[idx,:,sl:] = 1 # ToDo: more elegant way?
        self.mask = torch.from_numpy(self.mask).to(k_len.device, dtype=torch.bool).view(-1,ts)# BNxT

    def _attend(self, energy, value):
        attn = energy / self.temperature
        attn = attn.masked_fill(self.mask, -np.inf)
        attn = self.softmax(attn) # BNxT
        output = torch.bmm(attn.unsqueeze(1), value).squeeze(1) # BNxT x BNxTxD-> BNxD
        return output, attn

class ScaleDotAttention(BaseAttention):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, num_head):
        super().__init__(temperature, num_head)

    def forward(self, q, k, v):
        ts = k.shape[1]
        energy = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) # BNxD * BNxDxT = BNxT
        output, attn = self._attend(energy,v)
        
        attn = attn.view(-1,self.num_head,ts) # BNxT -> BxNxT

        return output, attn


class LocationAwareAttention(BaseAttention):
    ''' Location-Awared Attention '''
    def __init__(self, kernel_size, kernel_num, dim, num_head, temperature):
        super().__init__(temperature, num_head)
        self.prev_att  = None
        self.loc_conv = nn.Conv1d(num_head, kernel_num, kernel_size=2*kernel_size+1, padding=kernel_size, bias=False)
        self.loc_proj = nn.Linear(kernel_num, dim,bias=False)
        self.gen_energy = nn.Linear(dim, 1)
        self.dim = dim

    def reset_mem(self):
        super().reset_mem()
        self.prev_att = None

    def set_mem(self, prev_att):
        self.prev_att = prev_att

    def forward(self, q, k, v):
        bs_nh,ts,_ = k.shape
        bs = bs_nh//self.num_head

        # Uniformly init prev_att
        if self.prev_att is None:
            self.prev_att = torch.zeros((bs,self.num_head,ts)).to(k.device)
            for idx,sl in enumerate(self.k_len):
                self.prev_att[idx,:,:sl] = 1.0/sl

        # Calculate location context
        loc_context = torch.tanh(self.loc_proj(self.loc_conv(self.prev_att).transpose(1,2))) # BxNxT->BxTxD
        loc_context = loc_context.unsqueeze(1).repeat(1,self.num_head,1,1).view(-1,ts,self.dim)   # BxNxTxD -> BNxTxD
        q = q.unsqueeze(1) # BNx1xD
        
        # Compute energy and context
        energy = self.gen_energy(torch.tanh( k+q+loc_context )).squeeze(2) # BNxTxD -> BNxT
        output, attn = self._attend(energy,v)
        attn = attn.view(bs,self.num_head,ts) # BNxT -> BxNxT
        self.prev_att = attn

        return output, attn

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
            return results, loss
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
