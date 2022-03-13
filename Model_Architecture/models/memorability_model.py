import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

''' train features to approximate memorability score (regression) '''

class H_MLP(nn.Module):
    '''
        MLP model for handcrafted features
    '''
    def __init__(self, model_config):

        super(H_MLP, self).__init__()
        self.input_size = model_config["sequential_input_size"] + model_config["non_sequential_input_size"]
        self.Linear_1 = nn.Linear(self.input_size, model_config["hidden_size"])
        self.Linear_2 = nn.Linear(model_config["hidden_size"],32)
        self.Relu = nn.ReLU()
        self.BatchNorm = nn.BatchNorm1d(model_config["hidden_size"])
        self.Linear_3 = nn.Linear(32, 1)
        self.Sigmoid = nn.Sigmoid()
        self.Relu_2 = nn.ReLU()
        self.Dropout = nn.Dropout(p=model_config["dropout_rate"])

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
        x = self.Dropout(x)
        x = self.Relu(x)
        x = self.Linear_2(x)
        x = self.Dropout(x)
        x = self.Relu(x)
        # x = self.BatchNorm(x)
        x = self.Linear_3(x)
        x = self.Dropout(x)
        # predictions = self.Relu_2(x)
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

        # self.Sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

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

        predictions = self.ReLU(out)
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

        self.GruLayerF = nn.Sequential(nn.BatchNorm1d(model_config["fc_1"]["input_size"]),
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
        # (N, C, H, W) = (16, 1, 64, 64)
        out = self.convBlock1(inp)
        # (N, C, H, W) = (16, 4, 32, 32)
        out = self.convBlock2(out)
        # (N, C, H, W) = (16, 8, 16, 16)
        out = self.convBlock3(out)
        # (N, C, H, W) = (16, 16, 8, 8)
        out = out.contiguous().view(out.size()[0], out.size()[-1], -1)
        # [N, 8, 128]
        out, _ = self.GruLayer(out)
        # [N, 8, 32]
        out = out.contiguous().view(out.size()[0],  -1)
        # [N, 256]

        out = self.GruLayerF(out)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        return out

class BidirectionalLSTM(nn.Module):
    """ source: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py """
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        # ks = [3, 3, 3, 3, 3, 3, 2]
        # ps = [1, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1]
        # nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        # N, C, H, W

        # N, 1, 32, 128
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def create_msg(self):
        # Messages for user
        msg = []
        msg.append('Model spec.| CRNN: use CRNN model for melspectrogram inputs(img)')
        return msg

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

class E_Transformer(nn.Module):
    '''
        use Transformer encoder (visual transformer) for melspectrogram inputs, ref:
        https://arxiv.org/pdf/1706.03762.pdf
    '''
    def __init__(self, model_config):
        super(E_Transformer, self).__init__()
        self.d_model = model_config["transformer_encoder"]["d_model"]
        self.nhead = model_config["transformer_encoder"]["nhead"]
        self.dim_feedforward = model_config["transformer_encoder"]["dim_feedforward"]
        self.dropout = model_config["transformer_encoder"]["dropout"]
        self.seq_len = model_config["seq_len"]
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                    nhead=self.nhead,
                                                    dim_feedforward=self.dim_feedforward,
                                                    dropout=self.dropout)

        self.transforer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=model_config["transformer_encoder"]["num_layers"])
        self.transforer_encoder_f = nn.Sequential(nn.BatchNorm1d(self.seq_len), # after transpose
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
        msg.append('Model spec.| E_Transformer: use Transformer encoder for melspectrogram inputs(img)')
        return msg

    def forward(self, inp):

        # pad zero to seq_len (need to be divisible by n_head)
        # inp = F.pad(input=inp, pad=(0,  self.d_model - inp.size(-1), 0, 0), mode='constant', value=0)

        # (batch_size, 1(gray scale), embed_dim(n_mels), seq_len)
        out = inp.squeeze(1)
        # (batch_size, embed_dim(n_mels), seq_len)
        out = torch.transpose(out, 1, 2) # swap seq_len and embed_dim
        # (batch_size, seq_len, embed_dim(n_mels))
        out = self.transforer_encoder(out)

        out = self.transforer_encoder_f(out)
        out = out.contiguous().view(out.size()[0], -1)
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        return out


''' modified from https://github.com/santi-pdp/pase/blob/master/emorec/neural_networks.py '''  
class MLP(nn.Module):
    def __init__(self, options,inp_dim):
        super(MLP, self).__init__()
        
        self.input_dim=inp_dim
        self.dnn_lay=list(map(int, options['dnn_lay'].split(',')))
        self.dnn_drop=list(map(float, options['dnn_drop'].split(','))) 
        self.dnn_use_batchnorm=list(map(strtobool, options['dnn_use_batchnorm'].split(',')))
        self.dnn_use_laynorm=list(map(strtobool, options['dnn_use_laynorm'].split(','))) 
        self.dnn_use_laynorm_inp=strtobool(options['dnn_use_laynorm_inp'])
        self.dnn_use_batchnorm_inp=strtobool(options['dnn_use_batchnorm_inp'])
        self.dnn_act=options['dnn_act'].split(',')
        
       
        self.wx  = nn.ModuleList([])
        self.bn  = nn.ModuleList([])
        self.ln  = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
       
  
        # input layer normalization
        if self.dnn_use_laynorm_inp:
           self.ln0=LayerNorm(self.input_dim)
          
        # input batch normalization    
        if self.dnn_use_batchnorm_inp:
           self.bn0=nn.BatchNorm1d(self.input_dim,momentum=0.05)
           
           
        self.N_dnn_lay=len(self.dnn_lay)
             
        current_input=self.input_dim
        
        # Initialization of hidden layers
        
        for i in range(self.N_dnn_lay):
            
            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]))
            
            # activation
            self.act.append(act_fun(self.dnn_act[i]))
            
            
            add_bias=True
            
            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i],momentum=0.05))
            
            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias=False
            
                
            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i],bias=add_bias))
            
            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(torch.Tensor(self.dnn_lay[i],current_input).uniform_(-np.sqrt(0.01/(current_input+self.dnn_lay[i])),np.sqrt(0.01/(current_input+self.dnn_lay[i]))))
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]))
            
            current_input=self.dnn_lay[i]
             
        self.out_dim=current_input
         
    def forward(self, x):
        
        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            x=self.ln0((x))
        
        if bool(self.dnn_use_batchnorm_inp):

            x=self.bn0((x))
        
        for i in range(self.N_dnn_lay):
           
            if self.dnn_use_laynorm[i] and not(self.dnn_use_batchnorm[i]):
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))
          
            if self.dnn_use_batchnorm[i] and not(self.dnn_use_laynorm[i]):
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))
           
            if self.dnn_use_batchnorm[i]==True and self.dnn_use_laynorm[i]==True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))
          
            if self.dnn_use_batchnorm[i]==False and self.dnn_use_laynorm[i]==False:
                x = self.drop[i](self.act[i](self.wx[i](x)))
            
          
        return x

class LSTM(nn.Module):
    
    def __init__(self, options,inp_dim):
        super(LSTM, self).__init__()
        
        self.input_dim=inp_dim
        self.hidden_size=int(options['hidden_size'])
        self.num_layers=int(options['num_layers'])
        self.bias=bool(strtobool(options['bias']))
        self.batch_first=bool(strtobool(options['batch_first']))
        self.dropout=float(options['dropout'])
        self.bidirectional=bool(strtobool(options['bidirectional']))
        self.out_dim=self.hidden_size+self.bidirectional*self.hidden_size
        
        self.lstm = nn.ModuleList()
        
        for i in range(self.num_layers):
            input_size = self.input_dim if i == 0 else self.out_dim
            self.lstm.append(nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, batch_first=True, 
                                        bias=self.bias,dropout=self.dropout,bidirectional=self.bidirectional))
        
         
        self.linear = nn.Linear(self.out_dim, 1)

        self.relu = nn.ReLU()
               
        
    def forward(self, x):

        hidden_states_list = []                 
        for idx, layer in enumerate(self.lstm):
            inputs = x if idx == 0 else hidden_states

            # in shape: (batch_size, seq_len, input_size)
            hidden_states, _ = layer(inputs)
            # out shape: (batch_size, seq_length, hidden_size*bidirectional)
            
            hidden_states_list.append(hidden_states)

        # use last time step
        out = hidden_states[:, -1, :]
        out = self.linear(out)

        predictions = self.relu(out)
                
        return predictions, hidden_states_list

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def act_fun(act_type):

 if act_type=="relu":
    return nn.ReLU()
            
 if act_type=="tanh":
    return nn.Tanh()
            
 if act_type=="sigmoid":
    return nn.Sigmoid()
           
 if act_type=="leaky_relu":
    return nn.LeakyReLU(0.2)
            
 if act_type=="elu":
    return nn.ELU()
                     
 if act_type=="softmax":
    return nn.LogSoftmax(dim=1)
        
 if act_type=="linear":
     return nn.LeakyReLU(1) # initializzed like this, but not used in forward!

def strtobool (val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return 1
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return 0
    else:
        raise ValueError("invalid truth value %r" % (val,))

def context_window(fea,left,right):
 
    N_elem=fea.shape[0]
    N_fea=fea.shape[1]
    
    fea_conc=np.empty([N_elem,N_fea*(left+right+1)])
    
    index_fea=0
    for lag in range(-left,right+1):
        fea_conc[:,index_fea:index_fea+fea.shape[1]]=np.roll(fea,lag,axis=0)
        index_fea=index_fea+fea.shape[1]
        
    fea_conc=fea_conc[left:fea_conc.shape[0]-right]
    
    return fea_conc


if __name__ == "__main__":
    # model = MLP()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sequential_input_size = 408
    non_sequential_input_size = 2
    model = H_LSTM(sequential_input_size, non_sequential_input_size, hidden_size=640, num_layers=4, device=device, bidirectional=True).to(device)
