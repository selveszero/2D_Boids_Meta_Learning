import torch
from torch import nn

# decentralized model, each agent is treated independently
class BaselineModel(nn.Module):
    def __init__(self, device):
        super(BaselineModel, self).__init__()
        self.device = device
        self.hidden_size = 3

        self.fc_h_1 = nn.Linear(4+self.hidden_size, 128)
        self.fc_o_1 = nn.Linear(4+self.hidden_size, 128)

        self.fc_h_2 = nn.Linear(128, self.hidden_size)
        self.fc_o_2 = nn.Linear(128, 4)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x, h):

        # combine the state and hidden state
        combine = torch.cat((x, h), dim=1)

        # hidden branch
        h = self.tanh(self.fc_h_1(combine))
        h = self.fc_h_2(h)
        # output branch
        o = self.tanh(self.fc_o_1(combine))
        o = self.fc_o_2(o)

        return o, h

    def initHidden(self, n_agent):
        return torch.zeros(n_agent, self.hidden_size).to(self.device)