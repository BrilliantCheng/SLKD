The code is built on mdistiller.

Installation
Environments:

- Python 3.8
- PyTorch 1.7.0
- torchvision 0.8.0

Install the package:

- sudo pip3 install -r requirements.txt
- sudo python3 setup.py develop

Training on CIFAR-100:
- Download the cifar_teachers.tar at https://github.com/megvii-research/mdistiller/releases/tag/checkpoints and untar it to './download_ckpts' via 'tar xvf cifar_teachers.tar'.

- python3 TECH_TRAIN.py --cfg configs/cifar100/Tech_kd.yaml 

Training on ImageNet
- Download the dataset at https://image-net.org/ and put them to ./data/imagenet
- python3 TECH_TRAIN.py --cfg configs/imagenet/r34_r18/Tech_kd.yaml

Acknowledgement
- Sincere gratitude to the contributors of mdistiller for your distinguished efforts.

More training details and hyper parameters will be released soon.