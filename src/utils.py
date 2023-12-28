import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD, Adam
from src.FM_DCN import FM_DCN

def train(args, data):
    predicts = list()
    model = FM_DCN(args, data)
    model.load_state_dict(torch.load('./resource/model/FM_DCN.pt'))
    model.eval()
    print(data)
    y_hat = model(torch.tensor(data['test_dataloader']))
    predicts.extend(y_hat.tolist())
    return predicts