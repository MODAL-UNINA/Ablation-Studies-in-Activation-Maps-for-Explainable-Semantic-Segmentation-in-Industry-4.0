#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# %%

def mDice(output, mask, smooth=1e-10, n_classes=2):
##    print(type(pred_mask))
##    print(pred_mask.shape)
##    print(type(mask))
##    print(mask.shape)
    torch.set_printoptions(profile='full')
    with torch.no_grad():
        output = F.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
##        print('SPEJAMA KAUKE1: ', output)
##        print('SPEJAMA KAUKE1: ', output.shape)
        output = output.contiguous().view(-1)
##        print('KAUKE: ', mask)
##        print(mask.shape)
        mask = torch.argmax(mask, dim=1)
##        print('KAUKE: ', mask)
##        print(mask.shape)
        mask = mask.contiguous().view(-1)
        

##        print('SPEJAMA KAUKE2: ', output)
##        print('SPEJAMA KAUKE1: ', output.shape)


        dice_per_class = []
        for clas in range(0, n_classes):

            true_class = output == clas
            true_label = mask == clas

            if true_label.long().sum().item() == 0: 
                dice_per_class.append(np.nan)
            else:
                intersect = torch.logical_and(true_class, true_label).sum().float().item()
                union = torch.logical_or(true_class, true_label).sum().float().item()

                dice = ((2*intersect) + smooth) / ((union +smooth)+intersect)
                dice_per_class.append(dice)
##        print(dice_per_class)
        
        return np.nanmean(dice_per_class)
