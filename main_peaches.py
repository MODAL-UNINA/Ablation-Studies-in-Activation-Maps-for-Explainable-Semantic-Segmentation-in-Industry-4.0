# %%

import cv2
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
import albumentations as A
import torch
import torch.nn as nn
import random
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
# import torchvision.transforms as transforms
from skimage.transform import resize
# import math
from PIL import Image
import random
import os
from model import UNet
from mDice import mDice

import torchvision.transforms.functional as TF

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#%%


SIZE = 192
THRESHOLD = 100
NAME = 'image196'

# %%

# getwd()
path = os.getcwd()
img = cv2.imread(path + '/peaches_file/peaches/pesca196.png')
msk = cv2.imread(path + '/peaches_file/masks/taglio196.png')
gray = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

if False:
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.imshow(msk)
    plt.subplot(1, 3, 3)
    plt.imshow(gray)
    plt.show()


# %%

# resizing
aug = A.Resize(height=SIZE, width=SIZE)
image = aug(image=img)['image']
mask = aug(image=gray)['image']
print(type(mask))

# %%
THRESHOLDS = [0.1]
MAPS_IND = []
for i in range(256):
    MAPS_IND.append(i)

mDice_lst = []
backgr_lgts = []
foregr_lgts = []

LGT_0 = []
LGT_1 = []
ctr = []

# %%

class myLayer(nn.Module):
        def __init__(self, size_in):
            super().__init__()
            self.size = size_in
            ind = 0

            aug = A.Resize(height=self.size, width=self.size)
            self.mask = aug(image = mask)['image']
            self.mask = torch.from_numpy(self.mask)

            self.masks = torch.empty((1, 256, 24, 24))

            CHANNELS = self.masks.shape[1]


            OCCL_COLOR = 0
            NOT_CLASS_OF_INTEREST = 30
            for i in range(self.size):
                for j in range(self.size):
                    self.mask[i][j] = OCCL_COLOR

    ##        IF WE USE THE SAME OCCLUSION ON EVERY FILTER/CHANNEL
    ##WE SHOULD ALSO TRY TO USE DIFFERENT OCCLUSIONS ON EACH CHANNEL
            self.masks[0][ind][:][:] = self.mask


        def forward(self, x):

            OCCL_COLOR = 0

            IMPORTANCE = [-0.06519858955117218, -0.058896181557636915, -0.04754865381604087, -0.0649361594798327, -0.06818528288614857, -0.05074049361105225, -0.050237881000372644, -0.03846015863491127, 0.0037127624574826703, -0.06091669636006932, -0.082547351213659, -0.06437210130458342, -0.04805217448925106, -0.053191051693594886, -0.039381236728436124, -0.06637960054407742, -0.06329665258085886, -0.06582674180670536, -0.054409141906499064, -0.06139532098556593, -0.031829410364691446, -0.06767661652526506, -0.07756004478004555, -0.030913099599452168, -0.07188979965155355, -0.06523619847431397, -0.03175752208102002, -0.06838399723659191, -0.07952849733072434, -0.05309434303408743, -0.058174877220841074, -0.06250398966353443, -0.04918853907440245, -0.057377371303353526, -0.06541743262104352, -0.04453714010522845, -0.04402597657238583, -0.05075244976770497, -0.06734842759233561, -0.06698421884569289, -0.0571312863575646, -0.028797540918818382, -0.06132222195185372, -0.04061998536391114, -0.053353065183427015, -0.0732908644643709, -0.02456044547922445, -0.05930783657150277, -0.04855652755311428, -0.07734823919478627, -0.06223104120121566, -0.04675031550790028, -0.06102498281684175, -0.05027934918926943, -0.05504077506839949, -0.05264916537846742, -0.05914476367538496, -0.06758210235020652, -0.06139115903230074, -0.03318091009771428, -0.05320126739706398, -0.09114148201876182, -0.04967117430940915, -0.056673547170271475, -0.040286650743308366, -0.05102222001116669, -0.07874181249172628, -0.04898717620824491, -0.05605523225881965, -0.06951196224333538, -0.06248703916296349, -0.06575606427307472, -0.050423731131632676, -0.03492431448456272, -0.06025789699412885, -0.05725402614294887, -0.06537074307259587, -0.06382711244247645, -0.05461186686645249, -0.022617721366912282, -0.06760518227285893, -0.07740930639996821, -0.033392488667340924, -0.07583888792519648, -0.04360683004264263, -0.058364435274101, -0.0630759177140488, -0.06401712452700166, -0.05085975249006927, -0.05600309433518848, -0.04221038121435577, -0.049894255004423224, -0.05323320092938961, -0.0626579819343464, -0.07024628214306962, -0.06426419320719873, -0.04745648546918634, -0.02386608033083601, -0.053680951428846276, -0.05151764379347915, -0.06933617647178936, -0.05498757773848263, -0.035040470816600236, -0.05188132283697903, -0.06129876366981357, -0.04050473709440421, -0.06003284883029851, -0.04929826329684831, -0.04935153629864272, -0.06816992149500616, -0.04708425550352345, -0.05202812627942384, -0.05458364125612676, -0.05762996403061166, -0.0648818270717526, -0.08204602502489773, 0.0547100082064831, -0.06445344857294845, -0.07958222436378404, -0.05479870073212065, -0.05304886423568056, -0.04695538629605772, -0.053617387051705226, -0.041324868903278876, -0.12042052055184853, -0.04099266936083933, -0.052382194994475, -0.06761032796053225, -0.05939062160554124, -0.034964496251541174, -0.06489726413477258, 0.019748087341353728, -0.06379192501941622, -0.060175187631967934, -0.07890276657527272, -0.04318253782522611, -0.07914582464595969, -0.049034092772325205, -0.06283596219034134, -0.06071889007215657, -0.04884256725024901, -0.06550876857724501, -0.05774506095636349, -0.08157247041519688, -0.08181878237661847, -0.03909746718762786, -0.0630955924022115, -0.06206380635183265, -0.015460221157074773, -0.050036745149847746, -0.029788691170953494, -0.025544860934257712, -0.07061230701477351, -0.07951336295521456, -0.05403660925332597, -0.01949792119932743, -0.026151900735954754, -0.04764400038175245, -0.052932253872377745, -0.05857487876556441, -0.06758535624094113, -0.06426373917593343, -0.0674766157529034, -0.04637172910452328, -0.05988801285666997, -0.05761490532697943, -0.06153228708392938, -0.058396217462671526, -0.05646666025705287, -0.029317860748844418, -0.099282489621102, -0.0386357930627022, -0.04065154053684902, -0.07081488063097184, -0.06713389781948456, -0.0536388778649291, -0.039674162566427806, -0.059329932759747045, -0.06063792116317928, -0.05861952517331824, -0.08689538162573958, -0.060064479675113934, -0.05105362384034948, -0.07919955167901939, -0.05583812964213194, -0.05863125431433832, -0.04722682132082552, -0.06359691859097279, -0.030822066330760876, -0.07076054822289175, -0.045562342702260554, -0.055117506352234044, -0.05773582898730253, -0.08100712581802927, -0.03751758972816253, -0.049151686870036154, -0.03612999450954888, -0.06219759423133906, -0.05322517971036943, -0.05016114971653809, -0.05599893238192329, -0.05831562691308198, -0.05316539892710582, -0.043823856987452794, -0.06480327966285689, -0.03780862376921549, -0.035505928535403346, -0.05917639452020038, -0.04852451834891111, -0.036109109071345395, -0.05357304333146159, -0.06656900725358224, -0.05379332416700635, -0.03272937600438017, -0.06799042780146024, -0.04597702459122837, -0.0423117058583937, -0.05992547043605666, -0.06354409962044368, -0.052268081803131305, -0.03515110310157669, -0.05666575296688394, -0.06908577822898014, -0.05797881138611195, -0.0861100588805374, -0.015228967899285423, -0.06051124644016247, -0.07393286467349552, -0.06435779931972668, -0.07889512371564028, -0.03283077632029566, -0.05287436488605286, -0.06681365443369774, -0.03881521108437057, -0.08093031886231718, -0.061999409584038566, -0.07215881317623979, -0.08130262449985762, -0.05815928881406601, -0.05004756622833723, -0.05466998286841002, -0.060707312274891595, -0.06275900389087413, -0.06232495000125381, -0.03611743297787577, -0.06928381153252554, -0.041849729045957844, -0.04765118921011959, -0.07429230609185265, -0.049681541356633344, -0.04815032091443194, -0.05708209963715783, -0.05120837282993692, -0.042690367933648256, -0.06985399912985628, -0.053818522902230125]

            for i in range(x.shape[1]):
                x[0,i,:,:] = x[0,i,:,:]*IMPORTANCE[i]
            new = torch.sum(x, dim=1).squeeze().detach().numpy()
            new = new/(len(IMPORTANCE))


            print('NEW SHAPE', new.shape)
            for i in range(new.shape[0]):
                for j in range(new.shape[1]):
                    if new[i, j] <= 0:
                        new[i, j] = 0

            print(new)
            
            plt.axis('off')
            plt.imshow(new, cmap='jet')
            plt.savefig('heatmap1.png', bbox_inches='tight', pad_inches=0)
            plt.show()


            new_resized  = resize(new, (192,192))
            
            plt.axis('off')
            plt.imshow(new_resized, cmap='jet')
            plt.savefig('heatmap2.png', bbox_inches='tight', pad_inches=0)
            plt.show()

            new_resized = torch.from_numpy(new_resized).unsqueeze(dim = 0)
            heatmap = torch.empty(1, 192, 192).uniform_(0, 255)

            heatmap = torch.cat((new_resized, new_resized))
            heatmap = torch.cat((heatmap, new_resized))
            print('shape', heatmap.shape)
            print(heatmap)
            
            plt.imshow(heatmap.permute(1,2,0)*255, cmap='viridis')
            plt.savefig('heatmap3.png', bbox_inches='tight', pad_inches=0)
            plt.show()
            

            img = cv2.imread(path + '/peaches_file/peaches/pesca196.png')
            im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_rgb = aug(image=im_rgb)['image']
            im_rgb = torch.from_numpy(im_rgb).permute(2,0,1).unsqueeze(dim=0).to(torch.float).to(device)/255
            
            plt.axis('off')
            plt.imshow(im_rgb.squeeze().permute(1,2,0))
            plt.imshow(new_resized.squeeze(), cmap='jet', alpha=0.5)
            plt.savefig('heatmap4.png', bbox_inches='tight', pad_inches=0)
            plt.show()
            
            
            img = TF.to_pil_image(im_rgb.squeeze())  # assuming your image in x
            h_img = TF.to_pil_image(heatmap)

            new_resized = TF.to_pil_image(new_resized)
            print(type(img))
            print(type(new_resized))


# %%

if __name__ == '__main__':

    model = torch.load(path + '/Unet-Mobilenet_v2_mIoU-0.778.pt', map_location=torch.device('cpu'))
    
    new_model = UNet(inp=3, init_features=32)
    
    new_state_dict = new_model.state_dict()
    model_state_dict = model.state_dict()
    new_state_dict = {k: v for k, v in model_state_dict.items() if k in new_state_dict}

    new_model.load_state_dict(new_state_dict)
    
    new_model.encoder4.enc4relu2 = myLayer(24)

    new_model.eval()
    new_model = new_model.to(device)

    output = new_model(torch.from_numpy(image).permute(2,0,1).unsqueeze(dim=0).to(torch.float).to(device)/255)


# %%

