#!/bin/bash

python feature_extract.py --device_id 0 --ckpt_name 'xlnet-1_144910.ckpt' --task 'BBBP' --save_path '100M_144910step'> ./log/0.log 2>&1 &
python feature_extract.py --device_id 1 --ckpt_name 'xlnet-1_144910.ckpt' --task 'BACE'  --save_path '100M_144910step'> ./log/1.log 2>&1 &
python feature_extract.py --device_id 2 --ckpt_name 'xlnet-1_144910.ckpt' --task 'ESOL'  --save_path '100M_144910step'> ./log/2.log 2>&1 &
python feature_extract.py --device_id 3 --ckpt_name 'xlnet-1_144910.ckpt' --task 'FreeSolv'  --save_path '100M_144910step'> ./log/3.log 2>&1 &
python feature_extract.py --device_id 4 --ckpt_name 'xlnet-1_144910.ckpt' --task 'Lipophilicity'  --save_path '100M_144910step'> ./log/4.log 2>&1 &
# python feature_extract.py --device_id 5 --ckpt_name 'xlnet-1_28982.ckpt' --task 'T_Genotoxicity'  --save_path '100M_28982step'> ./log/5.log 2>&1 &
# python feature_extract.py --device_id 6 --ckpt_name 'xlnet-1_28982.ckpt' --task 'T_H-HT'  --save_path '100M_28982step'> ./log/6.log 2>&1 &
# python feature_extract.py --device_id 7 --ckpt_name 'xlnet-1_28982.ckpt' --task 'T_HERG_Inhibitor'  --save_path '100M_28982step'> ./log/7.log 2>&1 &

