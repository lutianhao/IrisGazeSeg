import torch
import torchvision
import os

path1 = r'/root/LIKE/虹膜/data/train/boundary/iris/'
path2 = r'/root/LIKE/虹膜/tmp/Anno/trian/bp'

path_list = os.listdir(path1)

for name in os.listdir(path2):
    name_ = name.replace('json','png')
    if name_ not in path_list:
        print(name)
        os.remove(os.path.join(path2,name))