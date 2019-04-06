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

CUDA_VISIBLE_DEVICES=0 python tf_cnn_benchmarks.py \
--data_format=NCHW \
--batch_size=64 \
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
| bs=4  | 6GB  |
| bs=8  | 8GB  |
| bs=16  | 24GB  |
| bs=32  | 24GB  |
| bs=64  | 48GB |


**Throughput (samples/sec)** 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| bs=4  | 7.3 | 7.8  | 9.8  | 8.8 | 10.9  | 11.4 | 10.7  | 10.6  | 11.0  |
| bs=8  | OOM | 9.2 | 10.9  | 10.1  |  12.9 | 13.1  | 12.1  | 13.0  | 12.9  |
| bs=16 |  OOM | OOM | OOM  | OOM  | OOM  |  15.2 | 13.4  | 14.9  | 14.5  |
| bs=32  | OOM  | OOM  | OOM  | OOM  |  OOM | 16.3 | 13.9  | 15.6  | 15.1  |
| bs=64  | OOM  | OOM  | OOM  |  OOM | OOM  | OOM | OOM  | OOM  | 15.6  |

**Time cost**

# NasNet large configuration
https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.py#L61

batch_size = 16
total_training_steps = 250000

GPU: Titan RTX
Equation: (total_training_steps * batch_size / samples_per_second) / 3600

(250000 * 16 / 15.2) / 3600 = 73.09 hours
