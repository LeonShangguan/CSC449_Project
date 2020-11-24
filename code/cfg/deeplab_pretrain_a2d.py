from argparse import Namespace

#device = [0,1]
resume = ''
dataset = 'A2D'

# Training strategry" config to passed to a2d dataset
train = dict(
    batch_size=12,
    optimizer=dict(
        algorithm='SGD',
        args=dict(
            base_lr=0.0007,
            momentum=0.9,
            weight_decay=0.00004,
            policy='poly',
            learning_power=0.9,
            # policy='step',
            # rate_decay_factor=0.1,
            # rate_decay_step=2000,
            max_epoch=96)),
    data_list='train',
    crop_size=[224, 224],
    input_mean=[103.939, 116.779, 123.68],
    input_std=[1, 1, 1],
    rotation=[-10, 10],
    blur=True,
    crop_policy='random',
    flip=True,
    scale_factor=[0.5, 2.0],
    vis=False)
train = Namespace(**train)

# val strategry
val = dict(
    batch_size=12,
    data_list='val',
    input_mean=[103.939, 116.779, 123.68],
    input_std=[1, 1, 1],
    scale_factor=[1.0],
    vis=False)
val = Namespace(**val)

# Testing strategry
test = dict(
    batch_size=1,
    data_list='test',
    scale_factor=[1.0],
    input_mean=[103.939, 116.779, 123.68],
    input_std=[1, 1, 1],
    vis=False)
test = Namespace(**test)

# Logging
output_dir = 'logs/full_sup_a2d_lr7e4'
snapshot_pref = 'deeplab_pretrain_a2d'
logging = dict(log_dir='', print_freq=50, eval_freq=5)
