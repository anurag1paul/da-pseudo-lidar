cd /data/pseudo_lidar/pseudo_lidar-dev-dev

python3 da_model_adversarial.py --maxdisp 192 --model basic_adv --datapath /data/datasets --epochs 20 --btrain 16 --lr 0.0002 --lr_dis 0.0002 --lambda_adv_target 0.002 --iter_size 2 
#--loadmodel psmnet/trained_da_adv/finetune_1.tar --loadcritic psmnet/trained_da_adv/finetune_critic1.tar

sleep 1h