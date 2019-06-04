




import os,pickle,time,sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 25})
plt.rcParams["font.family"] = "Times New Roman"

dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))


from metric import evaluate


def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


class Generator(nn.Module):
    def __init__(self, nc):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(nc * 64, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 10),

        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, nc * 64),
            nn.Tanh(),
        )

    def forward(self, input):
        input=input.view(input.shape[0],-1)
        z = self.encoder(input)
        output = self.decoder(z)
        output=output.view(output.shape[0],4,-1)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # input is (nc) x 64
            nn.Linear(nc * 64, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),

        )

        self.classifier=nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input=input.view(input.shape[0],-1)
        features = self.features(input)
        # features = self.feat(features.view(features.shape[0],-1))
        # features=features.view(out_features.shape[0],-1)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features







class BeatGAN_MOCAP:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

        self.batchsize = 64
        self.nz = 10
        self.niter = 300
        self.nc=4
        self.lr=0.0001
        self.beta1=0.5


        self.G = Generator( self.nc).to(device)
        self.G.apply(weights_init)


        self.D = Discriminator(self.nc).to(device)
        self.D.apply(weights_init)

        # Loss Functions
        self.bce_criterion = nn.BCELoss()
        self.l1_criterion=nn.L1Loss()
        self.mse_criterion=nn.MSELoss()

        self.optimizerD = optim.Adam(self.D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.total_steps = 0
        self.cur_epoch = 0


        self.real_label = 1
        self.fake_label = 0

        # -- Discriminator attributes.
        self.err_d_real = None
        self.err_d_fake = None

        # -- Generator attributes.
        self.err_g = None


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []




        print(" Train model ")
        best_auc=0
        best_auc_epoch=0


        for epoch in range(self.niter):
            self.cur_epoch+=1
            self.train_epoch()
            auc,th,f1=self.validate()
            if auc > best_auc:
                best_auc = auc
                best_auc_epoch=self.cur_epoch
                self.save_weight_GD()
            print("[{}] auc:{:.4f} th:{:.4f} f1:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc,th,f1,best_auc,best_auc_epoch ))




    def train_epoch(self):
        self.G.train()
        self.D.train()

        for train_x,train_y in self.dataloader["train"]:
            train_x=train_x.to(self.device)
            train_y=train_y.to(self.device)
            for i in range(1):
                self.D.zero_grad()
                batch_size=train_x.shape[0]
                self.y_real_, self.y_fake_ = torch.ones(batch_size).to(self.device), torch.zeros(batch_size).to(
                    self.device)
                # Train with real
                out_d_real, _ = self.D(train_x)
                # Train with fake
                fake = self.G(train_x)
                out_d_fake, _ = self.D(fake)
                # --

                self.err_d_real = self.bce_criterion(out_d_real,self.y_real_)
                self.err_d_fake = self.bce_criterion(out_d_fake,self.y_fake_)

                self.err_d = self.err_d_real + self.err_d_fake
                self.err_d.backward(retain_graph=True)
                self.optimizerD.step()


            self.G.zero_grad()

            _, feat_fake = self.D(fake)
            _, feat_real = self.D(train_x)


            self.err_g_adv = self.mse_criterion(feat_fake, feat_real)  # loss for feature matching
            self.err_g_rec = self.mse_criterion(fake, train_x)  # constrain x' to look like x

            self.err_g = self.err_g_rec+0.01*self.err_g_adv
            self.err_g.backward()
            self.optimizerG.step()


        # print(self.err_g.item())
        print("d_loss(R/F):{}/{}, g_loss:A/R{}/{}".format(self.err_d_real.item(),self.err_d_fake.item(),self.err_g_adv.item(),self.err_g_rec))
        if self.cur_epoch %50==0:
            fig, axarr = plt.subplots(2, 2, sharex=True, figsize=(6, 6))
            fake_ts=self.G(train_x).detach().cpu().numpy()
            input_ts=train_x.cpu().numpy()
            for r in range(2):
                axarr[r, 0].plot(np.transpose(input_ts[r],(1,0)))
                axarr[r, 0].set_ylim(-1, 1)

            for r in range(2):
                axarr[r, 1].plot(np.transpose(fake_ts[r],(1,0)))
                axarr[r, 1].set_ylim(-1, 1)
            # fig.savefig("./output/train/"+str(self.cur_epoch)+".png")



    def validate(self):
        y_, y_pred = self.predict(self.dataloader["test"])
        rocprc, rocauc, best_th, best_f1 = evaluate(y_, y_pred)
        return rocauc, best_th, best_f1


    def predict(self,dataloader_,scale=True):
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair=[]


            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            # self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
            #                             device=self.device)


            for i, data in enumerate(dataloader_, 0):
                test_x=data[0].to(self.device)
                test_y=data[1].to(self.device)

                fake = self.G(test_x)



                error = torch.mean(
                    torch.pow((test_x.view(test_x.shape[0], -1) - fake.view(fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.batchsize : i*self.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.batchsize : i*self.batchsize+error.size(0)] = test_y.reshape(error.size(0))
                # self.dis_feat[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0), :] = d_feat.reshape(
                #     error.size(0), self.opt.ndf*16*10)


            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_=self.gt_labels.cpu().numpy()
            y_pred=self.an_scores.cpu().numpy()
            # print(y_pred)

            return y_,y_pred


    def save_weight_GD(self):
        # save_dir=os.path.join("./output/","model")
        save_dir="model"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.G.state_dict(),
                   os.path.join(save_dir, 'G.pkl'))
        torch.save(self.D.state_dict(),
                   os.path.join(save_dir,'D.pkl'))

    def load(self):
        save_dir = "model"

        self.G.load_state_dict(
            torch.load(os.path.join(save_dir, 'G.pkl')))
        self.D.load_state_dict(
            torch.load(os.path.join(save_dir,'D.pkl')))

    def test_score_dist(self):
        self.load()
        self.G.eval()
        self.D.eval()
        y_, y_pred = self.predict(self.dataloader["test"])
        f = plt.figure()
        ax = f.add_subplot(111)
        X1=[]
        X2=[]
        for gt,sc in zip(y_,y_pred):
            if gt==0:
                X1.append(sc)
            else:
                X2.append(sc)


        _, bins, _ = ax.hist(X1, bins=55, range=[0, 1], density=True, alpha=0.3, color='r', label="walk")
        _ = ax.hist(X2, bins=bins, alpha=0.3, density=True, color='b', label="others")
        ax.set_yticks([])
        ax.set_xticks(np.arange(0,1.2, 0.2))
        ax.legend()
        f.savefig(os.path.join("./output", "dist.pdf"))
        auc_prc, roc_auc, best_threshold, best_f1=evaluate(y_, y_pred)
        print("ap:{}".format(auc_prc))
        print("auc:{}".format(roc_auc))
        print("best threshold:{}  ==> F1:{}".format(best_threshold,best_f1))



    def test_for_exp(self):

        self.load()
        self.G.eval()
        self.D.eval()
        test_0_cnt=0
        test_1_cnt=1
        legends=["lhumerus","rhumerus" , "lfemur", "rfemur"]
        with torch.no_grad():
            for i, data in enumerate(self.dataloader["test"], 0):
                data_x,data_y=data[0],data[1]
                gen_x=self.G(data_x.to(self.device))
                for  i in range(data_x.shape[0]):
                    input_x,input_y=data_x[i],data_y[i]
                    if input_y.item() ==1:
                        test_1_cnt+=1
                        if test_1_cnt>20:
                            return
                        fig, axarr = plt.subplots(1,2, figsize=(6, 6))
                        fake_ts = gen_x[i].cpu().numpy()
                        input_ts = input_x.cpu().numpy()
                        ax0=axarr[0].plot(np.transpose(input_ts, (1, 0)))
                        ax1=axarr[ 1].plot(np.transpose(fake_ts, (1, 0)))
                        fig.legend(iter(ax0), legends)
                        # ax1.legend(iter(ax1), legends)



                        fig.savefig("./output/test1/" + str(test_1_cnt) + ".png")
                    else:
                        test_0_cnt += 1
                        if test_0_cnt > 20:
                            return
                        fig, axarr = plt.subplots(1, 2, figsize=(6, 6))
                        fake_ts = gen_x[i].cpu().numpy()
                        input_ts = input_x.cpu().numpy()
                        ax0=axarr[0].plot(np.transpose(input_ts, (1, 0)))
                        ax1=axarr[1].plot(np.transpose(fake_ts, (1, 0)))

                        fig.legend(iter(ax0), legends)

                        fig.savefig("./output/test0/" + str(test_1_cnt) + ".png")


    def test_for_pic(self):
        from preprocess import load_for_pic
        self.load()
        self.G.eval()
        self.D.eval()
        walk_ts,run_ts=load_for_pic(ts_type="run")
        print(walk_ts.shape)
        input_walk=[]
        output_walk=[]
        walk_heat=[]

        input_run = []
        output_run = []
        run_heat = []

        legends = ["lhumerus", "rhumerus", "lfemur", "rfemur"]
        with torch.no_grad():
            #get res from walk
            for i in range(int(walk_ts.shape[1]/64)):
                ts=walk_ts[:,i*64:i*64+64]
                assert ts.shape[0]==4
                fake_ts=self.G(torch.from_numpy(np.array([ts])).float())
                fake_ts=fake_ts.numpy()[0]
                # heat_score=np.sqrt(np.sum((ts-fake_ts)**2,axis=0))
                heat_score=np.max((ts-fake_ts)**2,axis=0)
                input_walk.append(ts)
                output_walk.append(fake_ts)
                walk_heat.append(heat_score)
            input_walk=np.concatenate(input_walk, axis=1)
            output_walk=np.concatenate(output_walk, axis=1)
            walk_heat=np.concatenate(walk_heat)
            assert input_walk.shape[0]==4
            # get res from run
            for i in range(int(run_ts.shape[1] / 64)):
                ts = run_ts[:,i * 64:i * 64 + 64]
                assert ts.shape[0] == 4
                fake_ts = self.G(torch.from_numpy(np.array([ts])).float().to(self.device))
                fake_ts = fake_ts.numpy()[0]
                heat_score = np.sqrt(np.sum((ts - fake_ts) ** 2, axis=0))
                input_run.append(ts)
                output_run.append(fake_ts)
                run_heat.append(heat_score)

            input_run = np.concatenate(input_run, axis=1)
            output_run = np.concatenate(output_run, axis=1)
            run_heat = np.concatenate(run_heat)
            assert input_run.shape[0] == 4




        SIZE = 1
        for i in range(len(walk_heat / SIZE)):
            walk_heat[SIZE * i:SIZE * i + SIZE] = np.sum(walk_heat[SIZE * i:SIZE * i + SIZE]) / SIZE
        for i in range(len(run_heat / SIZE)):
            run_heat[SIZE * i:SIZE * i + SIZE] = np.sum(run_heat[SIZE * i:SIZE * i + SIZE]) / SIZE

        max_val=max(np.max(walk_heat),np.max(run_heat))
        min_val=min(np.min(walk_heat),np.min(run_heat))

        walk_heat_norm = np.reshape((walk_heat-min_val)/(max_val-min_val), (1, -1))
        run_heat_norm=np.reshape((run_heat-min_val)/(max_val-min_val),(1,-1))


        # for fig
        walk_points = np.arange(input_walk.shape[1])
        run_points = np.arange(input_run.shape[1])

        fig = plt.figure(figsize=(14,3 ))

        outer = gridspec.GridSpec(2, 3 ,wspace=0.2,height_ratios=[7, 1], width_ratios=[18,3,2])
        for i in range(6):
            if i == 0:
                inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                         subplot_spec=outer[i])
                ax = plt.Subplot(fig, inner[0])
                ax0 = ax.plot(walk_points, np.transpose(input_walk))
                ax.set_xlim((0, 210+128))
                ax.set_yticks([])
                ax.set_xticks([])
                ax.margins(0)
                fig.add_subplot(ax)


            elif i == 1:
                inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                         subplot_spec=outer[i], hspace=0.1)

                ax_inner_0 = plt.Subplot(fig, inner[0])
                start=210
                ax_inner_0.plot(run_points, np.transpose(input_walk)[start:start+run_points.shape[0],:])
                ax_inner_0.set_yticks([])
                ax_inner_0.set_xticks([])
                ax_inner_0.margins(0)
                fig.add_subplot(ax_inner_0)

                ax_inner_1 = plt.Subplot(fig, inner[1])
                ax_inner_1.plot(run_points, np.transpose(input_run))
                ax_inner_1.set_yticks([])
                ax_inner_1.set_xticks([])
                ax_inner_1.margins(0)
                fig.add_subplot(ax_inner_1)

            elif i == 2:
                inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                         subplot_spec=outer[i], hspace=0.1)

                ax_inner_0 = plt.Subplot(fig, inner[0])
                fig_walk= mpimg.imread("model/fig_walk.png")
                ax_inner_0.imshow(fig_walk,aspect="auto")
                ax_inner_0.set_yticks([])
                ax_inner_0.set_xticks([])
                fig.add_subplot(ax_inner_0)

                ax_inner_1 = plt.Subplot(fig, inner[1])
                fig_run=mpimg.imread("model/fig_run.png")
                ax_inner_1.imshow(fig_run,aspect="auto")
                ax_inner_1.set_yticks([])
                ax_inner_1.set_xticks([])
                fig.add_subplot(ax_inner_1)


            elif i == 3:
                inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                         subplot_spec=outer[i])

                ax = plt.Subplot(fig, inner[0])
                ax.imshow(walk_heat_norm, cmap="jet", aspect="auto",vmin = 0,vmax = 0.6)
                ax.set_yticks([])
                # ax.set_xticks([])
                fig.add_subplot(ax)
            elif i == 4:
                inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                         subplot_spec=outer[i])

                ax = plt.Subplot(fig, inner[0])
                ax.imshow(run_heat_norm, cmap="jet", aspect="auto",vmin = 0,vmax = 0.6)
                ax.set_yticks([])
                fig.add_subplot(ax)


        # fig.legend(iter(ax0), legends,loc="upper left")
        leg=fig.legend(iter(ax0),legends,bbox_to_anchor=(0.12, 0.985), loc='upper left', ncol=4,prop={'size': 12})
        leg.get_frame().set_alpha(0)

        # fig.tight_layout()
        # fig.show()
        # return
        fig.savefig("./output/pic_run.pdf")
        plt.clf()
        plt.close()



