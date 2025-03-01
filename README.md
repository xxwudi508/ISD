# MAEF: Multi Attention Edge Fusion Network for infrared sky regions detection
========

# Introduction
Link to article:
Link to code:

The network is designed for infrared sky region identification, used in conjunction with the IDS dataset mentioned in the article.
# Installation
List of required libraries and their versions.(For reference).
python-3.12.2

```
pytorch -- 2.3.1
numpy -- 1.26.4
opencv -- 4.10.0
matplotlib -- 3.8.4
timm -- 1.0.11
```

# Usage
- How to use the project, including example commands and expected outputs.

## Data preparation

Download and extract ISD Dataset from
(https://pan.baidu.com/s/1SeIlxZfivO74l3RljGUKfQ?pwd=ISDD).
google site：
https://drive.google.com/drive/folders/1rWaU0vsnQjxontKmw9yuxcnOyQtJ3vMR?usp=sharing

- We expect the directory structure to be the following:
```
./data/
  test/  # test images
    img/
    gt/
    test.txt     
  train/    # train images
    img/
    gt/
    train.txt  
    
```
## Train

First download the pre-trained model `swin_base_patch4_window12_384_22k.pth` from XX and put it in the `./pvt` folder

- Running the Train:
```
python train.py 
```
- After training, the result models will be saved in `./exp` folder
We need to put the result model into `out/swin` and name it `modelTest`.
- 
Run tests
```
python test.py 
```

## Acknowledgement
 The evaluation code comes from: https://github.com/zyjwuyan/SOD_Evaluation_Metrics.


## Contact
For any questions or issues, please contact:
liuxinlong@tyut.edu.cn


