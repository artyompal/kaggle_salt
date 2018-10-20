time python3.6 ./train.py --batch=12 --gpu=$1 --start_fold=$2 --cos_anneal \
    --lr_min=3e-3 --lr_max=1e-5 --epochs=90 --lr_rampdown=30 --mosaic=2
