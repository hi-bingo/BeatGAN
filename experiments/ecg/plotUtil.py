
import os
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({'font.size': 16})

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# sns.set_style('darkgrid')
# sns.set_palette('muted')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def save_plot_sample(samples, idx, identifier, n_samples=6, num_epochs=None,impath=None ,ncol=2):

    assert n_samples <= samples.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples.shape[2]

    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    if identifier=="ecg":
        for m in range(nrow):
            for n in range(ncol):
                sample = samples[n * nrow + m, 0, :]
                axarr[m, n].plot(x_points, sample, color=col)
                axarr[m, n].set_ylim(-1, 1)

    else:
        raise Exception("data type error:{}".format(identifier))

    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)

    assert impath is not  None
    fig.savefig(impath)
    plt.clf()
    plt.close()
    return


def save_plot_pair_sample(samples1,samples2, idx, identifier, n_samples=6, num_epochs=None,impath=None ,ncol=2):

    assert n_samples <= samples1.shape[0]
    assert n_samples % ncol == 0
    sample_length = samples1.shape[2] # N,C,L

    if not num_epochs is None:
        col = hsv_to_rgb((1, 1.0*(idx)/num_epochs, 0.8))
    else:
        col = 'grey'

    x_points = np.arange(sample_length)

    nrow = int(n_samples/ncol)
    fig, axarr = plt.subplots(nrow, ncol, sharex=True, figsize=(6, 6))
    if identifier=="ecg":
        for m in range(nrow):
            sample1=samples1[m,0,:]
            sample2=samples2[m,0,:]

            axarr[m,0].plot(x_points,sample1,color=col)
            axarr[m, 1].plot(x_points, sample2, color=col)
            axarr[m, 0].set_ylim(-1, 1)
            axarr[m, 1].set_ylim(-1, 1)

    else:
        raise Exception("data type error:{}".format(identifier))

    for n in range(ncol):
        axarr[-1, n].xaxis.set_ticks(range(0, sample_length, int(sample_length/4)))
    fig.suptitle(idx)
    fig.subplots_adjust(hspace = 0.15)

    assert  impath is not None

    fig.savefig(impath)
    plt.clf()
    plt.close()
    return


def plot_tsne(X,y,dim=2):
    tsne = TSNE(n_components=dim, verbose=1, perplexity=40, n_iter=1000)
    x_proj = tsne.fit_transform(X)
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f=plt.figure()
    if dim==2:
        ax = f.add_subplot(111)
        ax.scatter(x_proj[:, 0], x_proj[:, 1], lw=0, s=40,c=palette[y.astype(np.int)])
        ax.grid(True)
        for axi in (ax.xaxis, ax.yaxis):
            for tic in axi.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False

    elif dim==3:
        ax = Axes3D(f)
        ax.grid(True)
        ax.scatter(x_proj[:, 0], x_proj[:, 1],x_proj[:,2] ,lw=0, s=40,c=palette[y.astype(np.int)])
        for axi in (ax.xaxis, ax.yaxis,ax.zaxis):
            for tic in axi.get_major_ticks():
                tic.tick1On = tic.tick2On = False
                tic.label1On = tic.label2On = False
    f.savefig("sne.png")




def plot_dist(X1,X2,label1,label2,save_dir):
    assert  save_dir is not None
    f=plt.figure()
    ax=f.add_subplot(111)

    # bins = np.linspace(0, 1, 50)
    # _,bins=ax.hist(X1,bins=50)
    # print(bins)
    #
    # if logscale:
    #     bins = np.logspace(np.log10(bins[0]), np.log10(bins[1]), len(bins))

    _, bins, _ = ax.hist(X1, bins=50,range=[0,1],density=True,alpha=0.3,color='r', label=label1)
    _ = ax.hist(X2, bins=bins, alpha=0.3,density=True,color='b',label=label2)
    # ax.set_yticks([])
    ax.legend()
    f.savefig(os.path.join(save_dir, "dist"+label1+label2+".png"))

    #log scale figure
    f_log=plt.figure()
    ax_log=f_log.add_subplot(111)

    log_bins=np.logspace(np.log10(0.01),np.log10(bins[-1]),len(bins))
    _=ax_log.hist(X1, bins=log_bins, range=[0,1],alpha=0.3,density=True,color='r',label=label1)
    _ = ax_log.hist(X2, bins=log_bins,density=True, alpha=0.3,  color='b', label=label2)
    # ax_log.set_yticks([])

    ax_log.legend()
    ax_log.set_xscale('log')
    ax_log.set_xticks([round(x,2) for x in log_bins[::5]])
    ax_log.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_log.set_xticklabels([round(x,2) for x in log_bins[::5]], rotation=45)
    f_log.savefig(os.path.join(save_dir,"logdist"+label1+label2+".png"))




def save_pair_fig(input,output,save_path):
    '''
    save pair signal (current for first channel)
    :param input: input signal NxL
    :param output: output signal
    :param save_path:
    :return:
    '''
    save_ts_heatmap(input,output,save_path)

    # x_points = np.arange(input.shape[1])
    # fig, ax = plt.subplots(1, 2,figsize=(6, 6))
    # sig_in = input[ 0, :]
    # sig_out=output[0,:]
    # ax[0].plot(x_points, sig_in)
    # ax[1].plot(x_points,sig_out)
    # fig.savefig(save_path)
    # plt.clf()
    # plt.close()


def save_ts_heatmap(input,output,save_path):
    x_points = np.arange(input.shape[1])
    fig, ax = plt.subplots(2, 1, sharex=True,figsize=(6, 6),gridspec_kw = {'height_ratios':[6,1]})
    sig_in = input[0, :]
    sig_out = output[0, :]
    ax[0].plot(x_points, sig_in,'k-',linewidth=2.5,label="input signal")
    ax[0].plot(x_points,sig_out,'k--',linewidth=2.5,label="output signal")
    ax[0].set_yticks([])

    ax[0].legend(loc="upper right")



    heat=(sig_out-sig_in)**2
    heat_norm=(heat-np.min(heat))/(np.max(heat)-np.min(heat))
    heat_norm=np.reshape(heat_norm,(1,-1))

    ax[1].imshow(heat_norm, cmap="jet", aspect="auto")
    ax[1].set_yticks([])
    fig.tight_layout()
    # fig.show()
    # return
    fig.savefig(save_path)
    plt.clf()
    plt.close()



def loss_plot(hist, path = 'Train_hist.png', model_name = ''):
    x = range(len(hist['D_loss']))


    y1 = hist['D_loss']
    y2 = hist['G_loss']

    fig = plt.figure()

    ax1=fig.add_subplot(111)
    ax1.plot(x, y1,'r',label="D_loss")
    ax1.set_ylabel('D_loss')

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(x, y2, 'b',label="G_loss")

    ax2.set_ylabel('G_loss')

    ax2.set_xlabel('Iter')


    fig.legend(loc='upper left')

    ax1.grid(False)
    ax2.grid(False)

    # fig.tight_layout()

    path = os.path.join(path, model_name + '_loss.png')

    fig.savefig(path)
    # fig.show()




if __name__ == '__main__':
    import numpy as np
    foo = np.random.normal(loc=1, size=100)  # a normal distribution
    bar = np.random.normal(loc=-1, size=10000)  # a normal distribution
    max_val=max(np.max(foo),np.max(bar))
    min_val=min(np.min(foo),np.min(bar))
    foo=(foo-min_val)/(max_val-min_val)
    bar=(bar-min_val)/(max_val-min_val)
    plot_dist(foo,bar,"1","-1")


