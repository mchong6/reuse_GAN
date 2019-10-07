#!/bin/bash

GPU=7
for DIR in {trans_baseline_hinge_leaky3,trans_baseline_hinge_leaky_sobol3}
do
#for sample in {normal,sobol,halton}
for sample in {normal,sobol}
do
mkdir $DIR$sample
for seed in {0..500000..10000}
do
    echo $DIR $sample $seed
    CUDA_VISIBLE_DEVICES=$GPU python generate_images.py --image_size 64 --restore_gen $DIR --sampler $sample --skip $seed
    CUDA_VISIBLE_DEVICES=$GPU python fid.py ./pytorch_celeba_full.npz ./output/generated/temp/ --outdir $DIR$sample --seed $seed >> $DIR$sample/fid.txt
done
done
done
