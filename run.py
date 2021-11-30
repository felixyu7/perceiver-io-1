import torch
import numpy as np
import torch.optim as optim
from math import pi, log

import ROOT
from larcv import larcv

from perceiver.adapter import ImageInputAdapter, ClassificationOutputAdapter, SemanticSegOutputAdapter
from perceiver.model import PerceiverIO, PerceiverEncoder, PerceiverDecoder

from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.data import DataLoader, random_split

from einops import rearrange, repeat
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import matplotlib.pyplot as plt
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import accuracy

from uresnet import UResNet

batch_size = 4

class LArCVDataset(object):

    def __init__(self):
        self.files = ['/home/fyu04/perceiver-pytorch/larcv_p14_10000evts_5label_weight.root', '/home/fyu04/perceiver-pytorch/larcv_p11_10000evts_5label_weight.root']
        self.iocv =  larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickBackward)
        self.iocv.set_verbosity(5)
        #self.iocv.reverse_all_products() # Do I need this?
        self.iocv.add_in_file(self.files[0])
        self.iocv.add_in_file(self.files[1])
        self.iocv.initialize()
        # Get a list of opened files
        print('Listing files opened...')
        for count, name in enumerate(self.iocv.file_list()):
            print('  file {:d}: {:s}'.format(count,name))
        # Get number of entries in the file
        print('\nNumber of entries:', self.iocv.get_n_entries())
        # Get a list of data products stored
        print('\nListing data products stored...')
        ROOT.TFile.Open('/home/fyu04/perceiver-pytorch/larcv_p14_10000evts_5label_weight.root','READ').ls()

    def __len__(self):
        return self.iocv.get_n_entries()

    def __getitem__(self, idx):
        self.iocv.read_entry(idx)
        ev_wire    = self.iocv.get_data(larcv.kProductImage2D,"wire")
        img_v = ev_wire.Image2DArray()
        img_y = larcv.as_ndarray(img_v[2].as_vector())
        img_y = img_y.reshape((512, 512))
        img_y[img_y < 0.] = 0.

        ev_label    = self.iocv.get_data(larcv.kProductImage2D,"label")
        label_img = ev_label.Image2DArray()
        label_img_y = larcv.as_ndarray(label_img[2].as_vector())
        label_img_y[label_img_y >= 0] += 1
        label_img_y[label_img_y < 0] = 0
        label_img_y[label_img_y == 2] = 1
        label_img_y[label_img_y >= 3] = 2

        # label_img_y = label_img_y.reshape((512, 512))

        return (img_y, label_img_y)

class LAr_Perceiver(torch.nn.Module):
    def __init__(self):
        super(LAr_Perceiver, self).__init__()

        latent_shape = (32, 64)

        # Fourier-encode pixel positions and flatten along spatial dimensions
        input_adapter = ImageInputAdapter(image_shape=(512, 512, 1), num_frequency_bands=32)

        # Project generic Perceiver decoder output to specified number of classes
        output_adapter = ClassificationOutputAdapter(num_classes=3, num_outputs=512*512, num_output_channels=64)

        # Generic Perceiver encoder
        encoder = PerceiverEncoder(
            input_adapter=input_adapter,
            latent_shape=latent_shape,
            num_layers=3,
            num_cross_attention_heads=4,
            num_self_attention_heads=4,
            num_self_attention_layers_per_block=3,
            dropout=0.)

        # Generic Perceiver decoder
        decoder = PerceiverDecoder(
            output_adapter=output_adapter,
            latent_shape=latent_shape,
            num_cross_attention_heads=1,
            dropout=0.)

        # MNIST classifier implemented as Perceiver IO model
        self.perceiver = PerceiverIO(encoder, decoder)
        self.uresnet = UResNet(num_classes=3, input_channels=64, inplanes=16)

    def forward(self, x):
        x = x.reshape((batch_size, 512, 512, 1))
        mask = (x == 0.).reshape((batch_size, -1))
        x = self.perceiver(x, mask)
        # x = x.reshape(batch_size, -1, 512, 512)
        # x = self.uresnet(x)
        x = x.reshape(batch_size, 3, -1)
        return x

writer = SummaryWriter()

dataset = LArCVDataset()
model = LAr_Perceiver()

keep = []

for i in range(dataset.__len__()):
    if (dataset[i][0] > 0).sum() > 2621:
        keep.append(i)


dataset = Subset(dataset, keep)

num_entries = dataset.__len__()
print("num entries:", num_entries)

larcv_train, larcv_val, larcv_test = random_split(dataset, [num_entries - 1000, 1000, 0])
train_loader = DataLoader(larcv_train, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
val_loader = DataLoader(larcv_val, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5000, factor=0.1)
# scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=2500)

# inference mode
# checkpoint = torch.load("./ckpt/model_2000.ckpt")
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# model.eval()

total_iter = 0
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    model.cuda()
    model.train()
    for i, batch in enumerate(train_loader, 0):

        #scheduler_warmup.step(total_iter)
        # img, labels = dataset.__getitem__(1)
        img, labels = batch
        img = torch.as_tensor(img, dtype=torch.float32).cuda()

        labels = torch.as_tensor(labels, dtype=torch.long).cuda()

        # import matplotlib.patches as mpatches
        # data = labels.cpu().numpy().reshape(512,512)
        # values = np.unique(data.ravel())
        # plt.figure(figsize=(8,4))
        # im = plt.imshow(data, interpolation='none')
        #
        # # get the colors of the values, according to the
        # # colormap used by imshow
        # colors = [ im.cmap(im.norm(value)) for value in values]
        # # create a patch (proxy artist) for every color
        # patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
        # # put those patched as legend-handles into the legend
        # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        #
        # plt.grid(True)
        # plt.savefig("./imgs/event_%f.png" % i)

        optimizer.zero_grad()
        out = model(img)

        labels = labels.reshape(batch_size, -1)
        preds = out.argmax(axis=1).cpu()

        preds_flat = preds.flatten()
        labels_flat = labels.flatten().cpu()
        acc = 0.
        if torch.where(labels_flat > 0.)[0].shape[0] != 0:
            acc = accuracy(preds_flat[torch.where(labels_flat > 0.)], labels_flat[torch.where(labels_flat > 0.)])
        writer.add_scalar('train_acc', acc, total_iter)
        acc1 = 0.
        if torch.where(labels_flat == 1.)[0].shape[0] != 0:
            acc1 = accuracy(preds_flat[torch.where(labels_flat == 1.)], labels_flat[torch.where(labels_flat == 1.)])
        writer.add_scalar('train_acc1', acc1, total_iter)
        acc2 = 0.
        if torch.where(labels_flat == 2.)[0].shape[0] != 0:
            acc2 = accuracy(preds_flat[torch.where(labels_flat == 2.)], labels_flat[torch.where(labels_flat == 2.)])
        writer.add_scalar('train_acc2', acc2, total_iter)
        # acc3 = 0.
        # if torch.where(labels_flat == 3.)[0].shape[0] != 0:
        #     acc3 = accuracy(preds_flat[torch.where(labels_flat == 3.)], labels_flat[torch.where(labels_flat == 3.)])
        # writer.add_scalar('train_acc3', acc3, total_iter)
        # acc4 = 0.
        # if torch.where(labels_flat == 4.)[0].shape[0] != 0:
        #     acc4 = accuracy(preds_flat[torch.where(labels_flat == 4.)], labels_flat[torch.where(labels_flat == 4.)])
        # writer.add_scalar('train_acc4', acc4, total_iter)
        # acc5 = 0.
        # if torch.where(labels_flat == 5.)[0].shape[0] != 0:
        #     acc5 = accuracy(preds_flat[torch.where(labels_flat == 5.)], labels_flat[torch.where(labels_flat == 5.)])
        # writer.add_scalar('train_acc5', acc5, total_iter)
        # accb = 0.
        # if torch.where(labels_flat == 0.)[0].shape[0] != 0:
        #     accb = accuracy(preds_flat[torch.where(labels_flat == 0.)], labels_flat[torch.where(labels_flat == 0.)])
        # writer.add_scalar('train_accb', accb, total_iter)

        # inference mode
        # if epoch == 0:
        #     fig = plt.imshow(labels.cpu().numpy().reshape(512,512))
        #     plt.savefig("./output_infer/event_%f.png" % i)
        #     mask = (labels > 0).int().cpu().numpy().reshape(512,512)
        #     fig = plt.imshow(out.argmax(axis=1).cpu().numpy().reshape(512,512) * mask)
        #     plt.savefig("./output_infer/out_%f.png" % i)
        #
        # if (epoch % 2) == 0:
            # fig = plt.imshow(labels.cpu().numpy().reshape(512,512))
            # plt.savefig("./output_20events/event_%f.png" % i)
            # mask = (labels > 0).int().cpu().numpy().reshape(512,512)
            # fig = plt.imshow(out.argmax(axis=1).cpu().numpy().reshape(512,512) * mask)
            # plt.savefig("./output_20events/out_%f.png" % i)

        # if epoch % 50 == 0:
        #     fig = plt.imshow(out.argmax(axis=1).cpu().numpy().reshape(512,512))
        #     plt.savefig("./output_warmup/out_%f.png" % epoch)

        weights = torch.ones(3)
        weights[0] = 0.
        # weights[2] = 2.
        loss = F.cross_entropy(out.reshape(batch_size, -1, 512, 512), labels.reshape(batch_size, 512, 512), weights.cuda())
        # probs = F.softmax(out, dim=1)
        # loss = lovasz_softmax(probs.reshape(1,5,512,512), labels.reshape(1,512,512))
        print("loss: ", loss)

        writer.add_scalar('loss', loss, total_iter)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], total_iter)

        scheduler.step(loss)
        loss.backward()
        clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

        total_iter += 1

    # validation run
    with torch.no_grad():
        model.eval()
        vlosses = []
        vaccs = []
        for j, vbatch in enumerate(val_loader):
            vimg, vlabels = vbatch
            vimg = torch.as_tensor(vimg, dtype=torch.float32).cuda()
            vlabels = torch.as_tensor(vlabels, dtype=torch.long).cuda()
            vout = model(vimg)
            vlabels = vlabels.reshape(batch_size, -1)
            vpreds = vout.argmax(axis=1).cpu()

            vpreds_flat = vpreds.flatten()
            vlabels_flat = vlabels.flatten().cpu()
            if torch.where(vlabels_flat > 0.)[0].shape[0] != 0:
                vacc = accuracy(vpreds_flat[torch.where(vlabels_flat > 0.)], vlabels_flat[torch.where(vlabels_flat > 0.)])
                vaccs.append(vacc.item())

            vloss = F.cross_entropy(vout, vlabels, weights.cuda())
            vlosses.append(vloss.item())

        print("validation loss: ", np.mean(vlosses))
        writer.add_scalar('val_acc', np.mean(vaccs), total_iter)
        writer.add_scalar('validation_loss', np.mean(vlosses), total_iter)

torch.save({'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      }, "./ckpt/model_%d.ckpt" % epoch)
