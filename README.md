## DataSet
- For full MIT-BIH dataset, download from  
    https://www.dropbox.com/sh/2atveezs44lrwrd/AACaxZUmH_uw3M9ar_gD8iQPa?dl=0  （contains source and preprocessed data）
    and place them in experiments/ecg/dataset

- For motion capture dataset, in 
    experiments/mocap/dataset
    
## Usage
- For ecg full experiemnt (need to download full dataset)

    `sh run_ecg.sh`
    
- For ecg demo (there are demo data in experiments/ecg/dataset/demo )

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