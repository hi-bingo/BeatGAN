#!/bin/bash

cd experiments/ecg

test=1    # 0 means train the model, 1 means evaluate the model
threshold=0.02
fold_cnt=1

dataroot="./dataset/preprocessed/ano0/"
model="beatgan"

w_adv=1
niter=100
lr=0.0001
n_aug=0

outf="./output"

for (( i=0; i<$fold_cnt; i+=1))
do
    echo "#################################"
    echo "########  Folder $i  ############"
    if [ $test = 0 ]; then
	    python -u main.py  \
            --dataroot $dataroot \
            --model $model \
            --niter $niter \
            --lr $lr \
            --outf  $outf \
            --folder $i

	else
	    python -u main.py  \
            --dataroot $dataroot \
            --model $model \
            --niter $niter \
            --lr $lr \
            --outf  $outf \
            --folder $i  \
            --outf  $outf \
            --istest  \
            --threshold $threshold
    fi

done