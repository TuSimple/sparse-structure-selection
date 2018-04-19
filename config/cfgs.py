# mxnet version: https://github.com/huangzehao/incubator-mxnet-bk
mxnet_path = 'incubator-mxnet-bk/python/'
gpu_list = [0, 1, 2, 3]
dataset = "imagenet"
model_prefix = "resnet-50-sss-sgd-0.01"
network = "resnet"
depth = 50
model_load_prefix = model_prefix
model_load_epoch = 0
retrain = False
sss = True
gamma = 0.01

# data
data_dir = 'imagenet'
batch_size = 32
batch_size *= len(gpu_list)
kv_store = 'device'

# optimizer
lr = 0.1
wd = 0.0001
momentum = 0.9
if dataset == "imagenet":
    lr_step = [30, 60, 90]
else:
    lr_step = [120, 160, 240]
lr_factor = 0.1
begin_epoch = model_load_epoch if retrain else 0
num_epoch = 100
frequent = 50

# network config
if dataset == "imagenet":
    num_classes = 1000
if network.startswith("res"):
    units_dict = {"18": [2, 2, 2, 2],
                  "34": [3, 4, 6, 3],
                  "50": [3, 4, 6, 3],
                  "101": [3, 4, 23, 3],
                  "152": [3, 8, 36, 3]}
    units = units_dict[str(depth)]
    if depth >= 50:
        filter_list = [64, 256, 512, 1024, 2048]
        bottle_neck = True
    else:
        filter_list = [64, 64, 128, 256, 512]
        bottle_neck = False
    num_stage = 4