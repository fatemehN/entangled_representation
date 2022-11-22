import torch, cv2, os, glob
import numpy as np
import torchvision.transforms as transforms
from models.network_bmvae import MVAE
from PIL import Image
import torch.nn.functional as F

def _find_IoU(image, target, epsilon=1e-6):
    inter = torch.dot(image, target)
    sets_sum = torch.sum(image) + torch.sum(target)
    if sets_sum.item() == 0:
        sets_sum = 2*inter
    IoU = (inter + epsilon) / (sets_sum - inter + epsilon)
    return IoU

def _loading_single_video(path_im, path_mask, name):
    video_sequence, labels, names = [], [], []
    #Upload images 
    #Read images in the folder once
    image_set = sorted(glob.glob(path_im + '/' + name + '/*.jpg'))
    mask_set = sorted(glob.glob(path_mask + '/' + name + '/*.png'))
    episode_len = len(image_set) 
    for i in range(episode_len):
        frame = Image.open(image_set[i])
        mask = cv2.imread(mask_set[i])
        croped_frame, croped_msk = _net_input(frame, mask, 256)
        #Loading...
        video_sequence.append(croped_frame)
        labels.append(croped_msk)
        names.append(os.path.basename(image_set[i]))
    return video_sequence, labels, names


def _net_input(frame, mask, sfactor):
    mask = mask.astype(np.uint8).copy() 
    cnts, hierarchy = cv2.findContours(mask[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        dim = (sfactor, sfactor)
        im = cv2.resize(frame, dim, interpolation= cv2.INTER_LINEAR)
        msk = cv2.resize(mask, dim, interpolation= cv2.INTER_LINEAR)
    else:
        mask_points = np.concatenate(cnts, axis = 0)
        x, y, w, h = cv2.boundingRect(mask_points)
            
        padding = int((w+h)/4)
        x2 = min([x+w + padding, frame.shape[1]])
        y2 = min([y+h + padding, frame.shape[0]])
        x1 = max([0, x - padding])
        y1 = max([0, y - padding])

        img = frame[y1:y2, x1:x2, :]
        msk_crop = mask[y1:y2, x1:x2, :]
        
        dim = (sfactor, sfactor)
        im = cv2.resize(img, dim, interpolation= cv2.INTER_LINEAR)
        msk = cv2.resize(msk_crop, dim, interpolation= cv2.INTER_LINEAR)
    return np.asarray(im)/255, np.asarray(msk)/255


class Bmvae_Test():
    def __init__(self, Bmvae) -> None:
        super().__init__()
        self.Bmvae = Bmvae
        self.m = torch.nn.Softmax(dim=1)

    def bmvae_test(self, x):
        #Encoder
        x_encoded = self.Bmvae.encoder(x)
        #Bottleneck
        mu, sigma = self.Bmvae.fc_mu(x_encoded), self.Bmvae.fc_var(x_encoded)
        z = self.Bmvae.reparameterize_mv(mu, sigma).cpu()
        #Decoder
        x_hat = self.Bmvae.decoder(z)
        logits = self.Bmvae.outc(x_hat).float()
        #Calculate the class for each pixel 
        mask_pred = logits.argmax(dim=1).float()
        probs = self.m(logits)
        return mask_pred, probs, mu, sigma

    
    def distance_mvs(self, mu, sigma):
        """This function computes disctance for one estimation."""
        std_d = sigma.cpu().detach().numpy()
        x1 = np.mean(std_d[:, 0:50], axis = 0)
        x2 = np.mean(std_d[:, 50:100], axis = 0)
        x3 = np.mean(std_d[:, 100:150], axis = 0)
        x4 = np.mean(std_d[:, 150:200], axis = 0)
        x5 = np.mean(std_d[:, 200:250], axis = 0)
        X = np.array([x1, x2, x3, x4, x5]).reshape(5, -1) 
        cov_data = np.cov(X, rowvar=True) 
        cov_data = torch.FloatTensor(cov_data)
        cov_data = cov_data.clone()

        mu_data = torch.ones(32, 5)
        mu_data[:, 0] = torch.mean(mu[:, 0: 50], dim=1) #.detach()
        mu_data[:, 1] = torch.mean(mu[:, 50: 100], dim=1)
        mu_data[:, 2] = torch.mean(mu[:, 100: 150], dim=1)
        mu_data[:, 3] = torch.mean(mu[:, 150: 200], dim=1)
        mu_data[:, 4] = torch.mean(mu[:, 200: 250], dim=1)

        mu_target = torch.FloatTensor([1.065, 1.068, 0.0, 0.072, 0.083])
        mu_target = mu_target.repeat(32, 1)
        mu_target = mu_target.clone()
        cov_target = torch.FloatTensor([[0.265, -0.094, -0.014, -0.023, 0.124],
                                        [-0.094, 0.332, 0.0106, 0.056, -0.054],
                                        [-0.014, 0.011, 0.183, 0.0734, -0.2102],
                                        [-0.023, 0.056, 0.0734, 0.107, -0.028],
                                        [0.124, -0.054, -0.2102, -0.028, 0.564]])
        cov_target = cov_target.clone()

        first_term = (mu_data - mu_target) #.T #.transpose()
        second_term_inv = (cov_data + cov_target)/2
        seocnd_term = torch.inverse(second_term_inv)
        third_term = (mu_data - mu_target).T
        d_mah_1 = torch.matmul(first_term, seocnd_term)
        D_mah = torch.matmul(d_mah_1, third_term)
        return D_mah


    def _replace_prediction(self, mask, seg, sal):
        mask = mask.astype(np.uint8).copy() #.copy() #np.ascontiguousarray(mask, dtype=np.uint8)
        cnts, hierarchy = cv2.findContours(mask[:, :, 0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(cnts)
        if len(cnts) == 0:
            x1, y1, w, h = 0, 0, mask.shape[1], mask.shape[0]
        else:
            mask_points = np.concatenate(cnts, axis = 0)
            x, y, w, h = cv2.boundingRect(mask_points)
            padding = int((w+h)/4)
            x2 = min([x+w + padding, mask.shape[1]])
            y2 = min([y+h + padding, mask.shape[0]])
            x1 = max([0, x - padding])
            y1 = max([0, y - padding])
            w, h = x2 - x1, y2 - y1
        dim = (w, h)
        new_seg = cv2.resize(seg, dim, interpolation = cv2.INTER_AREA)
        new_sal = cv2.resize((255*sal).numpy(), dim, interpolation = cv2.INTER_AREA)
        final_map = np.zeros(mask.shape)
        smap = np.zeros(mask.shape[:2])
        final_map[y1:y1+h, x1:x1+w, :] = new_seg
        smap[y1:y1+h, x1:x1+w] = new_sal
        return final_map, smap


    def _find_best_match(self, output, mask, probs):
        #channel_mask = np.zeros((256, 256))
        best_IoU = 0
        for i in range(32):
            img = output[i, ...]
            binary_outs = F.one_hot(img.long(), num_classes=66).permute(2, 0, 1).float()
            mask = mask.float()
            for channel in range(1, 66):
                pmask = binary_outs[channel, ...].reshape(-1)
                single_IoU = _find_IoU(pmask, mask.reshape(-1))
                if single_IoU > best_IoU:
                    best_IoU = single_IoU
                    current_mask = binary_outs[channel, ...]
                    current_probs = probs[i, channel, ...]
        #Find a color for this object
        color_pix = [255, 255, 255]
        #Create colorful mask
        b_mask = np.zeros((256, 256, 3))
        b_mask[current_mask==1] = color_pix
        return b_mask, best_IoU, current_probs

    def predict(self, x, mask, name):
        original_msk = cv2.imread("./data/Annotations/" + name[:-4] + '.png')
        trans_labels = transforms.Compose([transforms.PILToTensor()])
        x = np.asarray(x)
        frm = np.tile(x, (32, 1, 1, 1))
        y_i = frm.transpose(0, 3, 1, 2)
        x_i = torch.from_numpy(y_i).float()
        msk = Image.fromarray(np.uint8(mask)[:, :, 0])   
        lbl = trans_labels(msk)
        with torch.no_grad():
            mask_pred, probs, mu, sigma = self.bmvae_test(x_i)
            mask_net = mask_pred.cpu().detach()
        segmentation_map, IoU, saliency_map = self._find_best_match(mask_net, lbl, probs)
        final_map, saliency_map = self._replace_prediction(original_msk, segmentation_map, saliency_map)
        print('Current IoU with GT is', IoU)
        d_mah = self.distance_mvs(mu, sigma)
        d_diag = (d_mah.numpy()).diagonal()
        return final_map, IoU, saliency_map, np.mean(d_diag)
