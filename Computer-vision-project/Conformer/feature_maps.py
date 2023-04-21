import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
from conformer import Conformer
from simclr import SimCLR


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

dog_image = Image.open(str('/nas/datahub/imagenet/val/n02110958/ILSVRC2012_val_00013278.JPEG'))

peacock_image = Image.open(str('/nas/datahub/imagenet/val/n01806143/ILSVRC2012_val_00028656.JPEG'))

bird_image = Image.open(str('/nas/datahub/imagenet/val/n01829413/ILSVRC2012_val_00008626.JPEG'))

bird2_image = Image.open(str('/nas/datahub/imagenet/val/n01829413/ILSVRC2012_val_00026845.JPEG'))


model = Conformer(patch_size=16, channel_ratio=4, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True)
checkpoint = torch.load('/nas/home/carlos/Conformer/output/Conformer_small_patch16_CIFAR10_lr1e-3_100epochs/checkpoint.pth')
model.load_state_dict(checkpoint['model'])

# encoder = Conformer()
# simclrmodel = SimCLR(encoder, 64, 1024)
# simclrmodel.load_state_dict(torch.load('/nas/home/carlos/SimCLR/imagenet_model/checkpoint_30.tar'))
# model = simclrmodel.encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

image = transform(peacock_image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")
image = image.to(device)

outputs = []
names = []

x_base = model.conv1(image)
names.append('Base Convolution')
outputs.append(x_base)
x = model.conv_1(x_base)
names.append('1st Convolutional Block')
outputs.append(x[0])

for i in range(2,13):
    x = eval('model.conv_trans_' + str(i) + '.cnn_block')(x[0])
    names.append(f'Conformer Block {i}')
    outputs.append(x[0])

print(len(outputs))
#print feature_maps
for feature_map in outputs:
    print(feature_map.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)
    
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i], cmap='inferno')
    a.axis("off")
    a.set_title(names[i], fontsize=30)
plt.savefig(str('feature_maps_peacock_simclr.jpg'), bbox_inches='tight')

attention_outputs = []
attention_names = []

y = model.trans_patch_conv(x_base).flatten(2).transpose(1,2)
print(y.shape)
for i in range(2,13):
    y = eval('model.conv_trans_' + str(i) + '.trans_block')(y)
    attention_names.append(f'Conformer Block {i}')
    attention_outputs.append(y.transpose(2,1).reshape(1,768,28,28))
    
for feature_map in attention_outputs:
    print(feature_map.shape)

processed = []
for feature_map in attention_outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)
    
fig = plt.figure(figsize=(30, 50))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    imgplot = plt.imshow(processed[i], cmap='inferno')
    a.axis("off")
    a.set_title(attention_names[i], fontsize=30)
plt.savefig(str('att_feature_maps_peacock_simclr.jpg'), bbox_inches='tight')