
import os
import numpy as np
import torch
from  torch.utils.data import DataLoader,TensorDataset



DATA_DIR="./dataset"


WALK=["35_01","35_02","35_03","35_04","35_05","35_06","35_07","35_08","35_09","35_10",
      "35_11","35_12","35_13","35_14","35_15","35_16"]

RUN=["35_17","35_18","35_19","35_20","35_21","35_22","35_23","35_24","35_25","35_26"]

WINDOW=64
STRIDE_TRAIN=5
STRIDE_TEST=20


def normalize(seq):
    '''
    normalize to [-1,1]
    :param seq:
    :return:
    '''
    return 2*(seq-np.min(seq))/(np.max(seq)-np.min(seq))-1

def read_mocap_file(file_path):
    timeseries=[]
    with open(file_path,"r") as f:
        for line in f.readlines():
            x=line.strip().split(" ")
            timeseries.append([float(xx) for xx in x])
    timeseries=np.array(timeseries)
    for i in range(timeseries.shape[1]):
        timeseries[:,i]=normalize(timeseries[:,i])

    return timeseries


def stat_data():
    cnt=0
    for walk in WALK:
        path=os.path.join(DATA_DIR, "walk", walk + ".amc.4d")
        ts=read_mocap_file(path)
        # print(ts.shape)
        cnt+=ts.shape[0]
    print (cnt)
    for run in RUN:
        path = os.path.join(DATA_DIR, "run", run + ".amc.4d")
        ts = read_mocap_file(path)
        # print(ts.shape)
        cnt += ts.shape[0]
    print(cnt)
    path = os.path.join(DATA_DIR, "other",  "49_02.amc.4d")
    ts = read_mocap_file(path)
    cnt += ts.shape[0]

    print(cnt)



def get_from_one(file_path,train=True):
    ts=read_mocap_file(file_path)
    ts_length=ts.shape[0]
    samples=[]
    stride=STRIDE_TRAIN if train else STRIDE_TEST
    for start in np.arange(0,ts_length,stride):
        if start+WINDOW>=ts_length:
            break
        samples.append(ts[start:start+WINDOW,:])
    # print(len(samples))
    # print(ts_length)
    # print(WINDOW)
    # print(stride)
    assert  len(samples)== np.ceil(((ts_length-WINDOW)/stride))
    return np.array(samples)


def load_data():
    batchsize=64
    train_x=None
    for walk in WALK[:-2]:
        ts=get_from_one(os.path.join(DATA_DIR,"walk",walk+".amc.4d"),train=True)
        if train_x is None:
            train_x=ts
        else:
            train_x=np.concatenate([train_x,ts])

    train_y=np.zeros([train_x.shape[0],1])


    test_x=None

    normal_test_cnt=0
    for walk in WALK[-2:]:
        ts = get_from_one(os.path.join(DATA_DIR, "walk", walk + ".amc.4d"), train=True)
        if test_x is None:
            test_x=ts
        else:
            test_x = np.concatenate([test_x, ts])
        normal_test_cnt+=ts.shape[0]

    for run in RUN[:]:
        ts = get_from_one(os.path.join(DATA_DIR, "run", run + ".amc.4d"), train=True)
        test_x = np.concatenate([test_x, ts])

    # add jump test data for experiment
    ts = get_from_one(os.path.join(DATA_DIR,"other","49_02.amc.4d"),train=True)
    test_x = np.concatenate([test_x, ts])

    test_y=np.ones([test_x.shape[0],1])
    test_y[:normal_test_cnt,:]=0

    train_x=np.transpose(train_x,(0,2,1))
    test_x=np.transpose(test_x, (0, 2, 1))
    print(train_x.shape)
    print(test_x.shape)
    # print(normal_test_cnt)
    # print(test_y)

    train_dataset = TensorDataset(torch.Tensor(train_x), torch.Tensor(train_y))
    test_dataset  = TensorDataset(torch.Tensor(test_x), torch.Tensor(test_y))


    dataloader = {"train": DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=batchsize,  # mini batch size
        shuffle=True,
        num_workers=0,
        drop_last=False),
        "test": DataLoader(
            dataset=test_dataset,  # torch TensorDataset format
            batch_size=batchsize,  # mini batch size
            shuffle=True,
            num_workers=0,
            drop_last=False),
    }
    return dataloader


def load_for_pic(ts_type="run"):
    walk_ts = read_mocap_file(os.path.join(DATA_DIR, "walk", WALK[-1] + ".amc.4d"))
    walk_ts = np.transpose(walk_ts)

    if ts_type=="run":
        run_ts = read_mocap_file(os.path.join(DATA_DIR, "run", RUN[1] + ".amc.4d"))
        run_ts = np.transpose(run_ts)
        ret_ts=run_ts
    elif ts_type=="jump":
        jump_ts=read_mocap_file(os.path.join(DATA_DIR, "other", "49_02.amc.4d"))
        jump_ts = np.transpose(jump_ts)
        ret_ts=jump_ts[:,600:750] #jump
        # ret_ts=jump_ts[:1500,1650]   #hop
    else:
        raise  Exception("ts type error!!!")
    return walk_ts,ret_ts




if __name__ == '__main__':
    # get_from_one(os.path.join(DATA_DIR,"run",RUN[0]+".amc.4d"))
    # load_data()
    stat_data()
    # ts1,ts2=load_for_pic(ts_type="jump")
    # print(ts1.shape)
    # print(ts2.shape)


