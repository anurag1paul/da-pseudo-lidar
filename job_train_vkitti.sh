cd /data/pseudo_lidar/pseudo_lidar-dev-dev

python3 da_model_mmd.py --maxdisp 192 --model basic_mmd --datapath /data/datasets --btrain 8 --lr 0.0001 --gamma 10 --k-critic 5

sleep 1h