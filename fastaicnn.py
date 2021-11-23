# %%
from fastprogress.fastprogress import master_bar
import torch
from torch import nn
from torchvision.utils import make_grid
from torchvision.models import resnet18

import matplotlib.pyplot as plt
from jupyterthemes import jtplot
from torch.utils.data import DataLoader
from fastai.vision.all import *
from fastai.vision.all import untar_data, URLs, ImageDataLoaders, cnn_learner, xresnet18, accuracy
path = "/home/CAMPUS/erza2018/cs152sp21-project.github.io/your_path"
files = get_image_files("/home/CAMPUS/erza2018/cs152sp21-project.github.io/your_path/train")
def label_func(f) : return f.split("._")[-1][:-4]
print(label_func(str(files[0])))

dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

#dls.show_batch()
learn = cnn_learner(dls, xresnet18, metrics=accuracy)
learn.fine_tune(1)

# # %%
learn.model
