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

CUDA_VISIBLE_DEVICES=1 python tf_cnn_benchmarks.py \
--data_format=NCHW \
--batch_size=4 \
--model=nasnetlarge \
--optimizer=momentum \
--variable_update=replicated \
--gradient_repacking=8 \
--num_gpus=1 \
--num_epochs=90 \
--weight_decay=1e-4 \
--data_dir=${DATA_DIR} \
--train_dir=${CKPT_DIR}
```

**Memory Requirement (MiB)**


| Batch Size  | Memory  |
|---|---|
| bs=4  | 10856  |
| bs=8  | 10856  |
| bs=16  | 19048  |
| bs=32  | 23432  |
| bs=64  |  |


**Throughput (samples/sec)** 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| bs=4  | 7.3 | 7.8  |   | 10.9  | 11.4 |   |   |   |
| bs=8  | OOM | 9.2 |   |  12.9 | 13.1  |   |   |   |
| bs=16 |  OOM | OOM |   | OOM  |  15.2 |   |   |   |
| bs=32  | OOM  | OOM  |   |  OOM | 16.3 |   |   |   |
| bs=64  | OOM  | OOM  |   | OOM  | OOM |   |   |   |