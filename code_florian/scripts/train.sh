time python3.6 ./train.py --batch=12 --gpu=0 --start_fold=1 --stop_fold=2 --cos_anneal --lr_min=1e-8 --lr_max=3e-2 --mosaic=2 --epochs=90 --lr_rampdown=30
time python3.6 ./train.py --batch=12 --gpu=0 --start_fold=4 --cos_anneal --lr_min=1e-8 --lr_max=3e-2 --mosaic=2 --epochs=90 --lr_rampdown=30
