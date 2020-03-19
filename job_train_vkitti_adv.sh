cd /data/pseudo_lidar/pseudo_lidar-dev-dev

python3 da_model_adversarial.py --maxdisp 192 --model basic_adv --datapath /data/datasets --epochs 20 --btrain 16 --lr 0.0001 --lr_dis 0.0001 --lr_scale 10 --lambda_adv_target 0.0001 --iter_size 1 
#--loadmodel psmnet/trained_da_adv/finetune_10.tar --loadcritic psmnet/trained_da_adv/finetune_critic10.tar --start_epoch 11

