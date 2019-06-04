

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import argparse
from preprocess import load_data
from model import  BeatGAN_MOCAP


device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")



if not os.path.exists("output"):
    os.mkdir("output")


dataloader=load_data()
print("load data success!!!")


model=BeatGAN_MOCAP(dataloader,device)


parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, default='train',help="mode: /train/eval ")
args = parser.parse_args()


if args.m=="train":
    model.train()

elif  args.m=="eval":
    model.test_score_dist()   #for score distribution. Fig 4

elif args.m=="pic":
    model.test_for_pic() # for case study on walk/run/jump. part of Fig 3

else:
    raise Exception("args error m:{}".format(args.m))


