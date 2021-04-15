import torch
from torch import nn

# decentralized model, each agent is treated independently
class BaselineModel(nn.Module):
    def __init__(self, device):
        super(BaselineModel, self).__init__()
        self.device = device
        self.hidden_size = 128
        self.input_size = 4

        self.gru_layers = 4
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.gru_layers)
        self.fc_o_1 = nn.Linear(self.hidden_size, 256)
        self.fc_o_2 = nn.Linear(256, self.input_size)
        self.fc_1 = nn.Linear(self.input_size, 256)
        self.fc_2 = nn.Linear(256, self.input_size)
        # self.fc_h_1 = nn.Linear(4+self.hidden_size, 128)
        # self.fc_o_1 = nn.Linear(4+self.hidden_size, 128)
        #
        # self.fc_h_2 = nn.Linear(128, 128)
        # self.fc_o_2 = nn.Linear(128, 128)
        #
        # self.fc_h_3 = nn.Linear(128, self.hidden_size)
        # self.fc_o_3 = nn.Linear(128, 4)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x, h):

        # combine the state and hidden state
        # combine = torch.cat((x, h), dim=1)

        # # hidden branch
        # h = self.tanh(self.fc_h_1(combine))
        # h = self.tanh(self.fc_h_2(h))
        # h = self.fc_h_3(h)
        # # output branch
        # o = self.tanh(self.fc_o_1(combine))
        # o = self.tanh(self.fc_o_2(o))
        # o = self.fc_o_3(o)

        o, h = self.gru(x, h)
        o = self.tanh(o)
        o = self.tanh(self.fc_o_1(o))
        o = self.fc_o_2(o)

        # x = x.squeeze(0)
        # o = self.tanh(self.fc_1(x))
        # o = self.fc_2(o)
        # o = o.unsqueeze(0)

        return o, h

    def initHidden(self, n_agent, bs):
        # return torch.zeros(n_agent, self.hidden_size).to(self.device)
        return torch.zeros(self.gru_layers, bs*n_agent, self.hidden_size).to(self.device)