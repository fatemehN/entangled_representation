import cv2, os, glob
import numpy as np
from models.network_bmvae import MVAE
import utils.bmvae as utl


if __name__ == '__main__':
    filename = "./ckpts/Bmvae_920.ckpt"
    bmvae = MVAE.load_from_checkpoint(filename)
    bmvae_test = utl.Bmvae_Test(bmvae)

    path_im = "./data/Samples/"
    path_mask = "./data/Annotations/"

    #Create output directory 
    directory = './Output/' 
    if not os.path.exists(directory):
        os.makedirs(directory)

    img_set = sorted(glob.glob(path_im + '/*.jpg'))
    msk_set = sorted(glob.glob(path_mask + '/*.png'))

    for i in range(len(img_set)):
        #Read the images
        img = cv2.imread(img_set[i])
        msk = cv2.imread(msk_set[i])
        name = os.path.basename(img_set[i])
        print(name)
        im, mask = utl._net_input(img, msk, 256)
        seg_map, IoUs, sal_map, d_mah = bmvae_test.predict(im, mask, name)

        #Resize the output to real image size
        h, w, k = img.shape

        final_map = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_LINEAR)
        saliency_map = cv2.resize(sal_map, (w, h), interpolation=cv2.INTER_LINEAR)

        #Save the binary mask
        cv2.imwrite(directory + name[:-4] + '_bmvae_mask.png' , (final_map).astype(np.uint8))

        #Save the Saliency map
        cv2.imwrite(directory + name[:-4] + '_bmvae_saliency.png' , (saliency_map).astype(np.uint8))
        print('D_mah for image', name, ':', d_mah)