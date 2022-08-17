# start wisdom
/home/ubuntu/anaconda3/envs/python38/bin/python -m visdom.server -p 50001&>/dev/null &

#train
python train.py --dataroot /data/cyclegan2/royalty/ --cuda --gpuid 0 --vis_env royalty --batchSize 1 --n_cpu 4 --output output/royalty --save_epoch_freq 20 --continue_train

