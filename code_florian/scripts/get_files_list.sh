#!/bin/bash

python3.6 get_files_list.py ../data/train/images/ >images_list.txt
python3.6 get_files_list.py ../data/train/masks/ >masks_list.txt
python3.6 get_files_list.py ../data/test/images/ >test_list.txt
md5sum images_list.txt 
md5sum masks_list.txt 
md5sum test_list.txt 

