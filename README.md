
## Overview
This is the implementation for the BeatGAN model architecture described in the paper: "BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series".

In this paper, we propose an unsupervised anomaly detection algorithm for time series data. BeatGAN has the following advantages: 1) Unsupervised: it is applicable even when labels are unavailable; 2) Effectiveness: It outperforms baselines in both accuracy and inference speed, achieving accuracy of nearly 0.95 AUC on ECG data and very fast inference (2.6 ms per beat); 3) Explainability: It pinpoints the time ticks involved in the anomalous patterns, providing interpretable output for visualization and attention routing; 4) Generality: BeatGAN also successfully detects unusual moions in multivariate motion-capture database.


## DataSet
- For full MIT-BIH dataset, download from  
    https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=0  （contain preprocessed data）
    and place them in experiments/ecg/dataset/preprocessed/

- For motion capture dataset, in 
    experiments/mocap/dataset
    
## Usage
- For ecg full experiemnt (need to download full dataset)

    `sh run_ecg.sh`
    
- For ecg demo (there are demo data in experiments/ecg/dataset/demo, the output dir is in experiments/ecg/output/beatgan/ecg/demo )

    `sh run_ecg_demo.sh`

- For motion experiment

    `sh run_mocap.sh`
    
## Require
- Python 3

### Packages
- PyTorch (1.0.0)
- scikit-learn (0.20.0)
- biosppy (0.6.1) # For data preprocess
- tqdm (4.28.1)
- matplotlib (3.0.2)


## Reference
If you find this code useful in your research, please, consider citing our paper:

``` latex
@inproceedings{zhou2019beatgan,
  title={BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time Series},
  author={Zhou, Bin and Liu, Shenghua and Bryan Hooi and Cheng, Xueqi and Ye, Jing },
  booktitle={International Joint Conference on Artificial Intelligence},
  year={2019},
}
```
