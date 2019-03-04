
import os
import numpy as np
import scipy.io as sio
# import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biosppy.signals import ecg
from tqdm import tqdm

ANO_RATIO=0   # add ANO_RATIO(e.g. 0.1%) anomalous to training data

LEFT=140
RIGHT=180



DATA_DIR="./dataset/source/"    # source mit-bih data
SAVE_DIR="./dataset/preprocessed/ano0/"


PATIENTS=[100,101,103,105,106,108,109,
          111,112,113,114,115,116,117,118,119,
          121,122,123,124,
          200,201,202,203,205,207,208,209,
          210,212,213,214,215,219,
          220,221,222,223,228,230,231,232,233,234] #remove 102 104 207 217
'''
As recommended by the AAMI, the records with paced beats(/) were not considered,
 namely 102, 104, 107, and 217.
'''

N={"N","L","R"}
S={"a", "J", "A", "S", "j", "e"}
V={"V","E"}
F={"F"}
Q={"/", "f", "Q"}
BEAT=N.union(S,V,F,Q)
ABNORMAL_BEAT=S.union(V,F,Q)


def bisearch(key, array):
    '''
    search value which is most closed to key
    :param key:
    :param array:
    :return:
    '''
    lo = 0
    hi = len(array)-1
    while lo <= hi:
        mid = lo + int((hi - lo) / 2)
        if key <array[mid]:
            hi = mid - 1
        elif key>array[mid] :
            lo = mid + 1
        else:
            return array[mid]
    if hi<0:
        return array[0]
    if lo>=len(array):
        return array[-1]
    return array[hi] if (key-array[hi])<(array[lo]-key) else array[lo]





def processPatient(patient):
    normal_samples = []
    abnormal_samples = []
    samples=[]
    record = sio.loadmat(os.path.join(DATA_DIR, str(patient) + ".mat"))
    annotation = sio.loadmat(os.path.join(DATA_DIR, str(patient) + "ann.mat"))
    if patient==114: # patient 114 record's mlii in lead B
        sig=record["signal"][:,1]
        sig2=record["signal"][:,0]
    else:           # others' in lead A
        sig = record["signal"][:,0]
        sig2=record["signal"][:, 1]
    assert len(sig)==len(sig2)
    sig_out=ecg.ecg(signal=sig, sampling_rate=360., show=False)

    sig=sig_out["filtered"]
    sig2=ecg.ecg(signal=sig2, sampling_rate=360., show=False)["filtered"]
    r_peaks=sig_out["rpeaks"]
    ts = record["tm"]
    ann_types = annotation["type"]
    ann_signal_idx = annotation["ann"]



    for ann_idx, ann_type in enumerate(ann_types):
        if ann_type in BEAT:
            sig_idx=ann_signal_idx[ann_idx][0]
            if sig_idx-LEFT>=0 and sig_idx+RIGHT<len(sig):
                if ann_type in N:
                    closed_rpeak_idx=bisearch(sig_idx,r_peaks)
                    if abs(closed_rpeak_idx-sig_idx)<10:
                        # normal_samples.append((sig[sig_idx-LEFT:sig_idx+RIGHT],ann_type))
                        samples.append(([sig[sig_idx-LEFT:sig_idx+RIGHT],sig2[sig_idx-LEFT:sig_idx+RIGHT]],'N',ann_type))
                else:
                    # abnormal_samples.append((sig[sig_idx-LEFT:sig_idx+RIGHT],ann_type))
                    AAMI_label=""
                    if ann_type in S:
                        AAMI_label = "S"
                    elif ann_type in V:
                        AAMI_label = "V"
                    elif ann_type in F:
                        AAMI_label = "F"
                    elif ann_type in Q:
                        AAMI_label="Q"
                    else:
                        raise  Exception("annonation type error")
                    assert AAMI_label!=""
                    samples.append(([sig[sig_idx - LEFT:sig_idx + RIGHT],sig2[sig_idx-LEFT:sig_idx+RIGHT]], AAMI_label, ann_type))



    return np.array(samples)



def process():
    Full_samples=[]
    N_samples=[]
    V_samples=[]
    S_samples=[]
    F_samples=[]
    Q_samples=[]

    print("Read from records")
    for patient in tqdm(PATIENTS):
        samples=processPatient(patient)
        for sample in samples:
            Full_samples.append(sample)
            if sample[1]=="N":
                N_samples.append(sample[0])
            elif sample[1]=="S":
                S_samples.append(sample[0])
            elif sample[1]=="V":
                V_samples.append(sample[0])
            elif sample[1]=="F":
                F_samples.append(sample[0])
            elif sample[1]=="Q":
                Q_samples.append(sample[0])
            else:
                raise  Exception("sample AAMI type error, input type: {}".format(sample[1]))

    Full_samples=np.array(Full_samples)
    N_samples=np.array(N_samples)
    V_samples=np.array(V_samples)
    S_samples=np.array(S_samples)
    F_samples=np.array(F_samples)
    Q_samples=np.array(Q_samples)

    np.random.shuffle(N_samples)
    np.random.shuffle(V_samples)
    np.random.shuffle(S_samples)
    np.random.shuffle(F_samples)
    np.random.shuffle(Q_samples)





    if ANO_RATIO >0:
        print("before shapes")
        print("N \t:{}".format(N_samples.shape))
        print("V \t:{}".format(V_samples.shape))
        print("S \t:{}".format(S_samples.shape))
        print("F \t:{}".format(F_samples.shape))
        print("Q \t:{}".format(Q_samples.shape))


        N_size=N_samples.shape[0]
        V_size=V_samples.shape[0]
        S_size=S_samples.shape[0]
        F_size=F_samples.shape[0]
        Q_size=Q_samples.shape[0]
        ANO_SIZE=V_size+S_size+F_size+Q_size

        ano_size=int(ANO_RATIO/(1+ANO_RATIO)*N_size)
        V_ano_size=int(ano_size*V_size/ANO_SIZE)
        S_ano_size = int(ano_size * S_size / ANO_SIZE)
        F_ano_size=  int(ano_size * F_size / ANO_SIZE)
        Q_ano_size = int(ano_size * Q_size / ANO_SIZE)
        if Q_ano_size==0:
            Q_ano_size=1

        N_samples=np.concatenate([N_samples,V_samples[-V_ano_size:]])
        V_samples=V_samples[:-V_ano_size]

        N_samples = np.concatenate([N_samples, S_samples[-S_ano_size:]])
        S_samples = S_samples[:-S_ano_size]

        N_samples = np.concatenate([N_samples, F_samples[-F_ano_size:]])
        F_samples = F_samples[:-F_ano_size]

        N_samples = np.concatenate([N_samples, Q_samples[-Q_ano_size:]])
        Q_samples = Q_samples[:-Q_ano_size]
        print("#########")




    print("N \t:{}".format(N_samples.shape))
    print("V \t:{}".format(V_samples.shape))
    print("S \t:{}".format(S_samples.shape))
    print("F \t:{}".format(F_samples.shape))
    print("Q \t:{}".format(Q_samples.shape))

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    np.save(os.path.join(SAVE_DIR, "full_samples.npy"), Full_samples)
    np.save(os.path.join(SAVE_DIR, "N_samples.npy"), N_samples)
    np.save(os.path.join(SAVE_DIR, "V_samples.npy"), V_samples)
    np.save(os.path.join(SAVE_DIR, "S_samples.npy"), S_samples)
    np.save(os.path.join(SAVE_DIR, "F_samples.npy"), F_samples)
    np.save(os.path.join(SAVE_DIR, "Q_samples.npy"), Q_samples)

    print("Finshed !!!")



if __name__ == '__main__':
    process()
