# Distributed (multi-gpu) adaptation of Contrastive Multiview Coding (CMC)

A distributed data-parallel adaptation of CMC, tested on AWS. Training methods below


1. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train_CMC.py --model resnet50v1 --batch_size 128 --data_folder CIFAR-10-images --model_path saved_models --tb_path tensorboard

2. python LinearProbing.py --dataset cifar10 --data_folder CIFAR-10-images --save_path saved_models --tb_path tensorboard --model_path ./saved_models/8g1024k/ckpt_epoch_240.pth --model resnet50v1 --learning_rate 0.1 --layer 5 --gpu 0

3. python extract_svm_features.py --dataset cifar10 --data_folder CIFAR-10-images --save_path saved_models --tb_path tensorboard --model_path ./saved_models/2048ksoftmax/ckpt_epoch_180.pth --model resnet50v1 --learning_rate 0.1 --layer 7 --gpu 0 --batch_size 256

4. CUDA_VISIBLE_DEVICES=2 python single_CMC.py --model resnet50v1 --batch_size 16 --data_folder CIFAR-10-images --model_path saved_models --tb_path tensorboard --save_freq 60 --nce_k 1024 --gpu 2



Some test records

8GPU on CIFAR10 k16384: 201.68 seconds per epoch
Interestingly, the value of k does not affect training time
1GPU on CIFAR10 k2048: 1101 seconds per epoch
