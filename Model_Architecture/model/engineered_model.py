import torch
import torch.nn as nn

FEATURE_NUM = 410

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.Linear_1 = nn.Linear(FEATURE_NUM, 640)
        self.Relu = nn.ReLU()
        self.Linear_2 = nn.Linear(640, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.Linear_1(inputs)
        x = self.Relu(x)
        x = self.Linear_2(x)
        prediction_prob = self.Sigmoid(x)
        return prediction_prob

if __name__ == "__main__":
    model = MLP()