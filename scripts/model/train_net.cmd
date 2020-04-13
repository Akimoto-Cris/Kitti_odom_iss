@echo off
set datadir=E:\\share_folder\\dataset
set n_fold=5
set batch_size=4
set epoch=100
set lr=0.0001
set lr_decay=30
set weight_decay=0.00005
set dropout=0.5
set num_workers=4
set grid_size=3
set model_path=weights_dropout_quat_balance_reg_rt_4\ckpt_epoch=2_1thfold_valxloss=0.290.pth
set save_dir=weights_dropout_quat_balance_reg_rt_4
set reg_lambda=0.005
set save_strategy=valxloss
set random_trans=1e-4

call conda activate torch
call python train_net.py -d %datadir% --n_fold %n_fold% -b %batch_size% --lr %lr% -wd %weight_decay% --weights_dir %save_dir%^
     --dropout %dropout% --num_workers %num_workers% --epoch %epoch% -gs %grid_size% --model_pth %model_path%^
     --reg_lambda %reg_lambda% --save_strategy %save_strategy% --lr_decay %lr_decay% -rt %random_trans%
     > \train_log.txt
     ::uncommend if you want to save output to log file
