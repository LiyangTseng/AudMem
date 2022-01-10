import torch
import torch.nn as nn
import torch.nn.functional as F

''' train features to approximate memorability score (regression) '''

class H_MLP(nn.Module):
    '''
        MLP model for handcrafted features
    '''
    def __init__(self, model_config):

        super(H_MLP, self).__init__()
        self.input_size = model_config["sequential_input_size"] + model_config["non_sequential_input_size"]
        self.Linear_1 = nn.Linear(self.input_size, model_config["hidden_size"])
        self.Relu = nn.ReLU()
        self.BatchNorm = nn.BatchNorm1d(model_config["hidden_size"])
        self.Linear_2 = nn.Linear(model_config["hidden_size"], 1)
        self.Sigmoid = nn.Sigmoid()

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

    def forward(self, features):
        
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

    def __init__(self, model_config):
        super().__init__()

        self.hidden_size = model_config["hidden_size"]
        self.num_layers = model_config["layer_num"]
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
            # in shape: (batch_size, seq_len, input_size)
            hidden_states, _ = layer(inputs)
            # out shape: (batch_size, seq_length, hidden_size*bidirectional)
    
        # only use the last timestep for linear input
        out = hidden_states[:, -1, :]
        out = torch.cat((out, non_sequential_features), 1)
        out = self.Batch_Norm_2(out)
        out = self.Linear(out)

        predictions = self.Sigmoid(out)
        return predictions


class E_CRNN(nn.Module):
    '''
        CRNN model for melspectrogram inputs, ref: https://github.com/XiplusChenyu/Musical-Genre-Classification
    '''
    def __init__(self, model_config):

        super(E_CRNN, self).__init__()
        cov1 = nn.Conv2d(in_channels=model_config["conv_1"]["in_channels"], out_channels=model_config["conv_1"]["out_channels"], kernel_size=model_config["conv_kernel_size"], stride=model_config["stride"], padding=model_config["padding"])
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(model_config["conv_1"]["out_channels"]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=model_config["conv_1"]["pool_kernel"]))

        cov2 = nn.Conv2d(in_channels=model_config["conv_2"]["in_channels"], out_channels=model_config["conv_2"]["out_channels"], kernel_size=model_config["conv_kernel_size"], stride=model_config["stride"], padding=model_config["padding"])
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(model_config["conv_2"]["out_channels"]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=model_config["conv_2"]["pool_kernel"]))

        cov3 = nn.Conv2d(in_channels=model_config["conv_3"]["in_channels"], out_channels=model_config["conv_3"]["out_channels"], kernel_size=model_config["conv_kernel_size"], stride=model_config["stride"], padding=model_config["padding"])
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(model_config["conv_3"]["out_channels"]),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=model_config["conv_3"]["pool_kernel"]))

        self.GruLayer = nn.GRU(input_size=model_config["gru"]["input_size"],
                               hidden_size=model_config["gru"]["hidden_size"],
                               num_layers=model_config["gru"]["layer_num"],
                               batch_first=True,
                               bidirectional=model_config["gru"]["bidirectional"])

        self.GruLayerF = nn.Sequential(nn.BatchNorm1d(model_config["gru"]["input_size"]),
                                       nn.Dropout(0.6))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=model_config["fc_1"]["input_size"], out_features=model_config["fc_1"]["output_size"]),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=model_config["fc_2"]["input_size"], out_features=model_config["fc_2"]["output_size"]),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=model_config["output"]["input_size"], out_features=model_config["output"]["output_size"]),
                                    nn.Sigmoid())    
    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| E_CRNN: use CRNN model for melspectrogram inputs(img)')
        # TODO: add one regarding attention
        # if self.encoder.vgg:
        #     msg.append('           | VCC Extractor w/ time downsampling rate = 4 in encoder enabled.')
        # if self.enable_ctc:
        #     msg.append('           | CTC training on encoder enabled ( lambda = {}).'.format(self.ctc_weight))
        # if self.enable_att:
        #     msg.append('           | {} attention decoder enabled ( lambda = {}).'.format(self.attention.mode,1-self.ctc_weight))
        return msg

    def forward(self, inp):
        # _input (batch_size, time, freq)

        out = self.convBlock1(inp)
        out = self.convBlock2(out)
        # 16, 32, 16, 16
        out = self.convBlock3(out)
        # 16, 64, 8, 8
        # [N, 256, 8, 8]
        out = out.contiguous().view(out.size()[0], out.size()[2], -1)
        # [N, 8, 2048]
        out, _ = self.GruLayer(out)
        # [N, 8, 256]
        out = out.contiguous().view(out.size()[0],  -1)
        # [N, 2048]

        out = self.GruLayerF(out)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        return out


if __name__ == "__main__":
    # model = MLP()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sequential_input_size = 408
    non_sequential_input_size = 2
    model = H_LSTM(sequential_input_size, non_sequential_input_size, hidden_size=640, num_layers=4, device=device, bidirectional=True).to(device)
