# %%
from fastai.vision.all import *

path = Path("/home/CAMPUS/erza2018/cs152sp21-project.github.io/PlantNet-300K/plantnet_300K/images")

dls = ImageDataLoaders.from_folder(path, train='train', valid='val', item_tfms=Resize(224))

#dls.show_batch()
#learn = cnn_learner(dls, xresnet18, metrics=accuracy)
#learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn = cnn_learner(dls, resnet50, metrics=accuracy)
learn.fine_tune(3)

# # %%
learn.model
