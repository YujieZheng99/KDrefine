# KDrefine


## Installation

This repo was tested with Ubuntu 20.04.4 LTS, Python 3.6, PyTorch 1.9.0, Torchvision 0.10.0 and CUDA 10.2.

| Package | Version (CIFAR) | Version (ImageNet) |

| ------ | ------ | ------ |

| h5py | 3.11.0 | 3.11.0 |
| lmdb | 1.4.1 | 1.4.1 |
| matplotlib | 3.5.3 | 3.9.0 |
| msgpack\_python | 0.5.6 | 0.5.6 |
| numpy | 1.21.6 | 1.21.6 |
| Pillow | 9.4.0 | 9.3.0 |
| pyarrow | 12.0.1 | 12.0.1 |
| scikit\_learn | 1.5.0 | 1.5.0 |
| seaborn | 0.13.2 | 0.13.2 |
| six | 1.16.0 | 1.16.0 |
| tensorboard\_logger | 0.1.0 | 0.1.0 |
| torch | 1.8.0+cu111 | 1.8.0+cu111 |
| torchvision | 0.9.0+cu111 | 0.9.0+cu111 |
| tqdm | 4.65.0 | 4.66.1 | 
```
pip install -r requirements.txt
```****

## Running

1. Fetch the pretrained teacher models:

    ```
    sh scripts/fetch_pretrained_teachers.sh
    ```
   which will download and save the models to `save/models`
   
2. An example of running KDrefine is given by:

    ```
    python train_student.py --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth --distill kdrefine --model_s represnet20  -r 0.1 -a 0.9 -b 0 --trial 1
    ```
    where the flags are explained as:
    - `--path_t`: specify the path of the teacher model
    - `--model_s`: specify the student model, see 'models/\_\_init\_\_.py' to check the available model types.
    - `--distill`: specify the distillation method
    - `-r`: the weight of the cross-entropy loss between logit and ground truth, default: `1`
    - `-a`: the weight of the KD loss, default: `None`
    - `-b`: the weight of other distillation losses, default: `None`
    - `--trial`: specify the experimental id to differentiate between multiple runs.
    
    
3. validation

Our models are at https://github.com/YujieZheng99/KDrefine/releases/tag/checkpoints

You can evaluate the performance of our models or models trained by yourself.

```
python validation.py --model represnet20 --model_path save/student_model/resnet110_2_resnet20_deploy_72.13.pth --blocktype ACTDB --deploy_flag True
```

4. (optional) Train teacher networks from scratch. Example commands are in `scripts/run_cifar_vanilla.sh`


## Benchmark Results on CIFAR-100:

We report the accuracy of last epoch and all results are average over 5 trials.

| Teacher <br> Student |WRN-40-2 <br> WRN-16-2 | WRN-40-2 <br> WRN-40-1 | resnet56 <br> resnet20 | resnet110 <br> resnet20 | resnet110 <br> resnet32 | vgg13 <br> vgg8  | vgg13 <br> mobilenetv2 | resnet50 <br> mobilenetv2 | resnet50 <br> vgg8 |
|:--------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:----------------------:|:-----------------------:|:----------------------:|:----------------------:|:-----------------------:|
| Teacher <br> Student | 75.61 <br> 73.26      |   75.61 <br> 71.98      |  72.34 <br> 69.06      |    74.31 <br> 69.06     |    74.31 <br> 71.14     |  74.64 <br> 70.36 |    74.64 <br> 64.60    |     79.34 <br> 64.60 |  79.34 <br> 70.36 |
|          KD          |    74.92         |      73.54          |          70.66          |          70.67          |          73.08          |       72.98       |       67.37          |           67.35           |       73.81        | 
|        FitNet        |    73.58        |          72.24          |         69.21          |          68.99          |          71.06          |       71.02       |       64.14          |           63.16           |       70.69        | 
|          AT          |    74.08         |       72.77        |         70.55          |          70.22          |          72.31          |       71.43       |       59.40          |           58.58           |       71.84        | 
|          SP          |    73.83        |          72.43          |          69.67          |          70.04          |          72.69          |       72.68       |       66.30          |           68.08           |       73.34        | 
|          CC          |      73.56        |        72.21        |        69.63          |          69.48          |          71.48          |       70.71       |       64.86          |           65.43           |       70.25        |  
|         VID          |      74.11         |         73.30          |        70.38          |          70.16          |          72.61          |       71.23       |       65.56          |           67.57           |       70.30        |  
|         RKD          |       73.35        |         72.22         |       69.61          |          69.25          |          71.82          |       71.48       |       64.52          |           64.43           |       71.50        | 
|         PKT          |       74.54        |        73.45         |       70.34          |          70.25          |          72.61          |       72.88       |       67.13          |           66.52           |       73.01        |
|          AB          |     72.50        |       72.38        |         69.47          |          69.53          |          70.98          |       70.94       |       66.06          |           67.20           |       70.65        | 
|          FT          |      73.25         |        71.59         |       69.84          |          70.22          |          72.37          |       70.58       |       61.78          |           60.99           |       70.29        |
|         FSP          |     72.91       |        \          |       69.95          |          70.11          |          71.89          |       70.23       |         \            |              \        |                 \      |
|         NST          |     73.68         |         72.24          |         69.60          |          69.53          |          71.96          |       71.53       |       58.16          |           64.96           |       71.28        |
|       CRD            |       75.48        |        74.14          |       71.16          |          71.46          |          73.48          |       73.94       |     **69.73**        |           69.11           |       74.30        |
| **KDrefine**         |    **75.97**          |          **74.62**          |        **71.81**        |        **71.99**        |        **74.08**        |     **74.70**     |       68.90          |         **69.60**         |     **74.88**      |

