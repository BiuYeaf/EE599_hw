from pytorch_model_summary import summary
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchviz import make_dot, make_dot_from_trace



class Model(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, output_size=3):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size*3, self.output_size)

    # create function to init state
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h = self.init_hidden(batch_size).to(device)

        out, h = self.rnn(x, h)

        avg_pool = F.adaptive_avg_pool1d(out.permute(0, 2, 1), 1).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool1d(out.permute(0, 2, 1), 1).view(x.size(0), -1)

        out = self.fc(torch.cat([h[-1], avg_pool, max_pool], dim=1))

        # return out, h
        return out

if __name__ == '__main__':
    model = Model(input_size=64, hidden_size=32, output_size=3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
# ############################
#     summary(
#         model,
#         torch.zeros(32, 1000, 64).to(device),
#         show_input=False,
#         print_summary=True,
#         show_hierarchical=True
#     )
#

    #
    #
    # ################### Visual
    x = torch.zeros(1, 1000, 64, dtype=torch.float, requires_grad=False)
    x= x.to(device)
    out = model(x)
    dot = make_dot(out)
    dot.format = 'png'
    dot.render("model")

