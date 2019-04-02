Installation
===

```bash
sudo pip3 install virtualenv
virtualenv -p /usr/bin/python3.6 venv-nasnet
. venv-nasnet/bin/activate

pip install -r requirements.txt 
```

Running NasNet Large with ImageNet Mini
===
**Download dataset and convert to TFRecord**

```bash
cd benchmarks
curl https://s3-us-west-2.amazonaws.com/lambdalabs-files/imagenet_mini.tar.gz | tar xvz -C .

```

**Train**

```bash
export DATA_DIR=`pwd`/imagenet_mini
mkdir ckpt
export CKPT_DIR=`pwd`/ckpt

cd scripts/tf_cnn_benchmarks/

CDUA_VISIBLE_DEVICES=0 python tf_cnn_benchmarks.py \
--data_format=NCHW \
--batch_size=64 \
--model=resnet50 \
--optimizer=momentum \
--variable_update=replicated \
--gradient_repacking=8 \
--num_gpus=1 \
--num_epochs=90 \
--weight_decay=1e-4 \
--data_dir=${DATA_DIR} \
--use_fp16 \
--train_dir=${CKPT_DIR}
```

**Memory Requirement (MiB)**


| Batch Size  | Memory  |
|---|---|
| bs=64  |   |
| bs=128  |   |
| bs=256  |   |
| bs=512  |  |
| bs=1024  |  |


**Throughput (samples/sec)** 

|   | 2060  | 2070  | 2080  |  2080 Ti | TitanRTX | V100 | Quadro RTX 6000 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|
| bs=64  | 3.86  | 3.92  |   | 5.48  | 5.75  |   |   |   |
| bs=128 |  4.48 | 4.50 |   | 6.45  |  6.92 |   |   |   |
| bs=256  | OOM  | OOM  |   | 7.04  |  7.84 |   |   |   |
| bs=512  | OOM  | OOM  |   |  OOM |  8.24 |   |   |   |
| bs=1024  | OOM | OOM  |   | OOM  |  OOM |   |   |   |