from functools import partial
import torch.nn as nn
import torch
from fastai.basic_train import Learner
from fastai.train import ShowGraph
from fastai.data_block import DataBunch
from torch import optim
from dataset.dataset import FracNetTrainDataset
import dataset.transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.model import UNet
from model.loss import MixLoss, DiceLoss


train_image_dir = 'data\\train_img'
train_label_dir = 'data\\train_label'
test_image_dir ='data\\test_img'
test_label_dir = 'data\\test_label'

batch_size = 1
num_workers = 0
optimizer = optim.SGD
criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)
model = UNet(1, 1, first_out_channels=16)
model = nn.DataParallel(model.cuda())
transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
thresh = 0.1
recall_partial = partial(recall, thresh=thresh)
precision_partial = partial(precision, thresh=thresh)
fbeta_score_partial = partial(fbeta_score, thresh=thresh)
ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
dl_train = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
ds_test = FracNetTrainDataset(test_image_dir, test_label_dir,
        transforms=transforms)
dl_test = FracNetTrainDataset.get_dataloader(ds_test, batch_size, False,
        num_workers)
databunch=DataBunch(dl_train, dl_test,
        collate_fn=FracNetTrainDataset.collate_fn)
learn = Learner(
        databunch,
        model,
        loss_func=criterion,
        metrics=dice
    )
learn.fit_one_cycle(
        30,
        1e-1,
        pct_start=0,
        div_factor=1000,
        callbacks=[
            ShowGraph(learn),
        ]
    )
torch.save(model.module.state_dict(), "./model.pth")


