mkdir -p logs
nohup python train.py \
    --config /home/chenyu/ADV/AD/config/cifar10.yaml\
    > logs/logs.out 2>&1 & 