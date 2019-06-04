## DataSet
- For full MIT-BIH dataset, download from  
    https://www.dropbox.com/sh/b17k2pb83obbrkn/AADzJigiIrottyTOyvAEU1LOa?dl=0  （contain preprocessed data）
    and place them in experiments/ecg/dataset

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