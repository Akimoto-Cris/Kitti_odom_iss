@echo off
set datadir=E:\\share_folder\\dataset
set n_fold=5
set batch_size=4
set epoch=50
set lr=0.0005
set weight_decay=0.00005
set dropout=0.3
set num_workers=4
set grid_size=3
set model_path=weights_dropout_quat_balance\ckpt_epoch=0_0thfold_trainloss=-5.551.pth
set save_dir=weights_dropout_quat_balance
set reg_lambda=0.
set save_strategy=trainloss

call conda activate torch
call python train_net.py -d %datadir% --n_fold %n_fold% -b %batch_size% --lr %lr% -wd %weight_decay% --weights_dir %save_dir%^
     --dropout %dropout% --num_workers %num_workers% --epoch %epoch% -gs %grid_size% --model_pth %model_path%^
     --reg_lambda %reg_lambda% --save_strategy %save_strategy%
     ::> %save_dir%\\train_log.txt
