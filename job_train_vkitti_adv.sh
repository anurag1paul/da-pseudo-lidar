cd /data/pseudo_lidar/pseudo_lidar-dev-dev

python3 da_model_adversarial.py --maxdisp 192 --model basic_adv --datapath /data/datasets --epochs 10 --btrain 8 --lr 0.0002 --lr_dis 0.0001 --lambda_adv_target 0.001 --iter_size 4

sleep 1h