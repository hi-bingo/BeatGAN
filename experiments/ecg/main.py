

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from options import Options

from data import load_data

# from dcgan import DCGAN as myModel


device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")




opt = Options().parse()
print(opt)
dataloader=load_data(opt)
print("load data success!!!")

if opt.model == "beatgan":
    from model import BeatGAN as MyModel

else:
    raise Exception("no this model :{}".format(opt.model))


model=MyModel(opt,dataloader,device)

if not opt.istest:
    print("################  Train  ##################")
    model.train()
else:
    print("################  Eval  ##################")
    model.load()
    model.test_type()
    # model.test_time()
    # model.plotTestFig()
    # print("threshold:{}\tf1-score:{}\tauc:{}".format( th, f1, auc))