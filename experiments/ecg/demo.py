import os
import  numpy as np

import torch
from  torch.utils.data import DataLoader,TensorDataset

from model import BeatGAN
from options import Options
import  matplotlib.pyplot as plt
import matplotlib
plt.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams.update({'font.size': 38})
from plotUtil import save_ts_heatmap
from data import normalize

device = torch.device("cpu")

SAVE_DIR="output/demo/"




def load_case(normal=True):
    if normal:
        test_samples = np.load(os.path.join("dataset/demo/", "normal_samples.npy"))
    else:
        test_samples = np.load(os.path.join("dataset/demo/", "abnormal_samples.npy"))

    for i in range(test_samples.shape[0]):
        for j in range(1):
            test_samples[i][j] = normalize(test_samples[i][j][:])
    test_samples = test_samples[:, :1, :]
    print(test_samples.shape)
    if not normal :
        test_y=np.ones([test_samples.shape[0],1])
    else:
        test_y = np.zeros([test_samples.shape[0], 1])
    test_dataset = TensorDataset(torch.Tensor(test_samples), torch.Tensor(test_y))

    return DataLoader(dataset=test_dataset,  # torch TensorDataset format
                      batch_size=64,
                      shuffle=False,
                      num_workers=0,
                      drop_last=False)

normal_dataloader=load_case(normal=True)
abnormal_dataloader=load_case(normal=False)
opt = Options()
opt.nc=1
opt.nz=50
opt.isize=320
opt.ndf=32
opt.ngf=32
opt.batchsize=64
opt.ngpu=1
opt.istest=True
opt.lr=0.001
opt.beta1=0.5
opt.niter=None
opt.dataset=None
opt.model = None
opt.outf=None



model=BeatGAN(opt,None,device)
model.G.load_state_dict(torch.load('model/beatgan_folder_0_G.pkl',map_location='cpu'))
model.D.load_state_dict(torch.load('model/beatgan_folder_0_D.pkl',map_location='cpu'))


model.G.eval()
model.D.eval()
with torch.no_grad():

    abnormal_input=[]
    abnormal_output=[]

    normal_input=[]
    normal_output=[]
    for i, data in enumerate(abnormal_dataloader, 0):
        test_x=data[0]
        fake_x, _ = model.G(test_x)

        batch_input = test_x.cpu().numpy()
        batch_output = fake_x.cpu().numpy()
        abnormal_input.append(batch_input)
        abnormal_output.append(batch_output)
    abnormal_input=np.concatenate(abnormal_input)
    abnormal_output=np.concatenate(abnormal_output)

    for i, data in enumerate(normal_dataloader, 0):
        test_x=data[0]
        fake_x, _ = model.G(test_x)

        batch_input = test_x.cpu().numpy()
        batch_output = fake_x.cpu().numpy()
        normal_input.append(batch_input)
        normal_output.append(batch_output)
    normal_input=np.concatenate(normal_input)
    normal_output=np.concatenate(normal_output)

    # print(normal_input.shape)
    # print(np.reshape((normal_input-normal_output)**2,(normal_input.shape[0],-1)).shape)

    normal_heat= np.reshape((normal_input-normal_output)**2,(normal_input.shape[0],-1))

    abnormal_heat = np.reshape((abnormal_input - abnormal_output)**2 , (abnormal_input.shape[0], -1))

    # print(normal_heat.shape)
    # assert False

    max_val = max(np.max(normal_heat), np.max(abnormal_heat))
    min_val = min(np.min(normal_heat), np.min(abnormal_heat))

    normal_heat_norm = (normal_heat - min_val) / (max_val - min_val)
    abnormal_heat_norm = (abnormal_heat - min_val) / (max_val - min_val)



    # for fig
    dataset=["normal","abnormal"]

    for d in dataset:
        if not os.path.exists(os.path.join(SAVE_DIR , d)):
            os.makedirs(os.path.join(SAVE_DIR , d))

        if d=="normal":
            data_input=normal_input
            data_output=normal_output
            data_heat=normal_heat_norm
        else:
            data_input = abnormal_input
            data_output = abnormal_output
            data_heat = abnormal_heat_norm

        for i in range(50):

            input_sig=data_input[i]
            output_sig=data_output[i]
            heat=data_heat[i]

            # print(input_sig.shape)
            # print(output_sig.shape)
            # print(heat.shape)
            # assert  False

            x_points = np.arange(input_sig.shape[1])
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 6), gridspec_kw={'height_ratios': [7, 1],
                                                                                   })

            sig_in = input_sig[0, :]

            sig_out = output_sig[0, :]
            ax[0].plot(x_points, sig_in, 'k-', linewidth=2.5, label="ori")
            ax[0].plot(x_points, sig_out, 'k--', linewidth=2.5, label="gen")
            ax[0].set_yticks([])

            # leg=ax[0].legend(loc="upper right",bbox_to_anchor=(1.06, 1.06))
            # leg.get_frame().set_alpha(0.0)

            heat_norm = np.reshape(heat, (1, -1))
            # heat_norm=np.zeros((1,320))
            # if d=="normal":
            #     heat_norm[0,100:120]=0.0003
            # else:
            #     heat_norm[0,100:120]=0.9997

            ax[1].imshow(heat_norm, cmap="jet", aspect="auto",vmin = 0,vmax = 0.2)
            ax[1].set_yticks([])
            # ax[1].set_xlim((0,len(x_points)))

            # fig.subplots_adjust(hspace=0.01)
            fig.tight_layout()
            # fig.show()
            # return
            fig.savefig(os.path.join(SAVE_DIR+d,str(i)+"_output.png"))

            fig2, ax2 = plt.subplots(1, 1)
            ax2.plot(x_points, sig_in, 'k-', linewidth=2.5, label="input signal")
            fig2.savefig(os.path.join(SAVE_DIR  + d, str(i) + "_input.png"))


            plt.clf()

print("output files are in:{}".format(SAVE_DIR))