""" Multi-Varitional UNET for Entanglement Representation Learning in video frames
    Fatemeh N. Nokabadi, June 2022
    """

from models.unet_parts import *
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from loss.IoU_score import iou_loss
from sklearn.utils import class_weight


class MV_Unet(pl.LightningModule):
    def __init__(self, n_channels=3, n_classes=66, bilinear=False, batch_size = 32):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.batch_size = batch_size

        self.save_hyperparameters()
        torch.autograd.set_detect_anomaly(True)

        #Define encoder layers
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 =Down(512, 1024)

        #Flatten Layer
        #self.flat = nn.Linear(524288, 512)
        self.flat = nn.Linear(262144, 1024)
        #self.flat = nn.Linear(65536, 1024)

        #Add Bottleneck Layers 
        self.fc_mu = nn.Linear(1024, 250)
        self.fc_var = nn.Linear(1024, 250)
        self.z_dim = nn.Linear(250, 1024*64)

        #Unflatten Layer
        #self.unflat = nn.Linear(, )

        #Define Decoder layers
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr= 0.0001)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.00005, max_lr=0.0001, cycle_momentum=False)
        return [optimizer], [scheduler]    
    

    def training_step(self, batch):
        x, labels = batch
        labels = labels.squeeze(1)
        mask_labels = labels.cpu().detach()

        total_weights = np.zeros((self.n_classes,))
        lbls = np.unique(mask_labels)
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(mask_labels), y = np.ravel(labels.cpu()))
        for i in range(len(class_weights)):
            total_weights[lbls[i]] = class_weights[i]
        class_weights = torch.from_numpy(total_weights).to('cuda')

        #Encoder
        x = x.to('cuda', dtype = torch.float)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #Bottleneck
        x5 = torch.flatten(x5, 1)
        x5 = self.flat(x5)
        mu = self.fc_mu(x5)
        co = self.fc_var(x5)
        z = self.reparameterize_mv(mu, co)
        z_dec = self.z_dim(z)
        z_dec = z_dec.view(self.batch_size, 1024, 8, 8)
        
        #Decoder
        x = self.up1(z_dec, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x).float()
 
        prob = F.softmax(logits, dim=1).float()
        mask_pred = logits.argmax(dim=1).float()
        
        target = F.one_hot(labels, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        binary_out = F.one_hot(mask_pred.long(), num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        # construction loss
        cons_loss = iou_loss(binary_out, target) 
        ce_criterion = nn.CrossEntropyLoss(class_weights.float())
        ce_loss = ce_criterion(logits, labels)
        seg_loss = ce_loss + cons_loss
        
        # kl
        kld_loss = self._kl_loss_mv(mu, co)

        # total_loss
        total_loss = seg_loss + 20*kld_loss 
        
        self.log_dict({
            'total_loss': total_loss,
            'kl': kld_loss,
            'iou_loss': cons_loss,
            'ce_loss': ce_loss,
        },sync_dist=True) #'kl': kld_loss,

        return total_loss

    def validation_step(self, batch, batch_id):
        x, labels = batch
        labels = labels.squeeze(1)
        mask_labels = labels.cpu().detach()

        total_weights = np.zeros((self.n_classes,))
        lbls = np.unique(mask_labels)
        class_weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(mask_labels), y = np.ravel(labels.cpu()))
        for i in range(len(class_weights)):
            total_weights[lbls[i]] = class_weights[i]
        class_weights = torch.from_numpy(total_weights).to('cuda')


        #Encoder
        x = x.to('cuda', dtype = torch.float)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #Bottleneck
        x5 = torch.flatten(x5, 1)
        x5 = self.flat(x5)
        mu = self.fc_mu(x5)
        co = self.fc_var(x5)
        z = self.reparameterize_mv(mu, co)
        z_dec = self.z_dim(z)
        z_dec = z_dec.view(self.batch_size, 1024, 8, 8)
        
        #Decoder
        x = self.up1(z_dec, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x).float()

        prob = F.softmax(logits, dim=1).float()
        mask_pred = logits.argmax(dim=1).float()

        target = F.one_hot(labels, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        binary_out = F.one_hot(mask_pred.long(), num_classes=self.n_classes).permute(0, 3, 1, 2).float()

        # construction loss
        cons_loss = iou_loss(binary_out, target) 
        #cross entropy 
        ce_criterion = nn.CrossEntropyLoss(class_weights.float())
        ce_loss = ce_criterion(logits, labels)
        seg_loss = ce_loss + cons_loss
        
        # kl
        kld_loss = self._kl_loss_mv(mu, co)

        # total_loss        
        total_loss = seg_loss + 20*kld_loss 

        self.log_dict({
            'valid_seg': seg_loss,
            'valid_kl': kld_loss,
            'valid_loss': total_loss,
        }, sync_dist=True)

        return logits
    def reparameterize_mv(self, mu, logvar):
        """
        This function map data into the target prior.
        """
        std_d = logvar.cpu().detach().numpy()

        x1 = np.mean(std_d[:, 0:50], axis = 0)
        x2 = np.mean(std_d[:, 50:100], axis = 0)
        x3 = np.mean(std_d[:, 100:150], axis = 0)
        x4 = np.mean(std_d[:, 150:200], axis = 0)
        x5 = np.mean(std_d[:, 200:250], axis = 0)

        X = np.array([x1, x2, x3, x4, x5]).reshape(5, -1) 
        
        cov_data = np.cov(X, rowvar=True) 
        cov_data = torch.FloatTensor(cov_data)
        cov_data = cov_data.to('cuda').clone()

        std_data = torch.ones(5)
        std_data[0] = torch.sqrt(cov_data[0][0])
        std_data[1] = torch.sqrt(cov_data[1][1])
        std_data[2] = torch.sqrt(cov_data[2][2])
        std_data[3] = torch.sqrt(cov_data[3][3])
        std_data[4] = torch.sqrt(cov_data[4][4])
        std_data = std_data.to('cuda')

        mu_data = torch.ones(self.batch_size, 5)
        mu_data[:, 0] = torch.mean(mu[:, 0: 50], dim=1) #.detach()
        mu_data[:, 1] = torch.mean(mu[:, 50: 100], dim=1)
        mu_data[:, 2] = torch.mean(mu[:, 100: 150], dim=1)
        mu_data[:, 3] = torch.mean(mu[:, 150: 200], dim=1)
        mu_data[:, 4] = torch.mean(mu[:, 200: 250], dim=1)
        mu_data = mu_data.to('cuda')

        a_d = mu_data - std_data
        b_d = mu_data + std_data

        mu_target = torch.FloatTensor([1.065, 1.068, 0.0, 0.072, 0.083])
        mu_target = mu_target.repeat(self.batch_size, 1)
        mu_target = mu_target.to('cuda').clone()

        cov_target = torch.FloatTensor([[0.265, -0.094, -0.014, -0.023, 0.124],
                                        [-0.094, 0.332, 0.0106, 0.056, -0.054],
                                        [-0.014, 0.011, 0.183, 0.0734, -0.2102],
                                        [-0.023, 0.056, 0.0734, 0.107, -0.028],
                                        [0.124, -0.054, -0.2102, -0.028, 0.564]])
        cov_target = cov_target.to('cuda').clone()
        p = torch.distributions.multivariate_normal.MultivariateNormal(mu_target, cov_target)
        eps = p.sample((50,))
        eps = eps.view(self.batch_size, 250)
        eps2 = eps.clone()
        
        alpha1, beta1= (std_data[0]/0.52), mu_data[:, 0] - (std_data[0]/0.52)*1.060
        alpha2, beta2= (std_data[1]/0.54), mu_data[:, 1] - (std_data[1]/0.54)*1.060
        alpha3, beta3= (std_data[2]/0.43), mu_data[:, 2] - (std_data[2]/0.43)*0
        alpha4, beta4= (std_data[3]/0.34), mu_data[:, 3] - (std_data[3]/0.34)*0.0751
        alpha5, beta5= (std_data[4]/0.74), mu_data[:, 4] - (std_data[4]/0.74)*0.0873

        eps2[:, 0: 50] = alpha1 * eps[:, 0: 50] + torch.tile(beta1, (50, 1)).T
        eps2[:, 50: 100] = alpha2 * eps[:, 50: 100] + torch.tile(beta2, (50 ,1)).T
        eps2[:, 100: 150] = alpha3 * eps[:, 100: 150] + torch.tile(beta3, (50, 1)).T
        eps2[:, 150: 200] = alpha4 * eps[:, 150: 200] + torch.tile(beta4, (50, 1)).T
        eps2[:, 200:250] = alpha5 * eps[:, 200:250] + torch.tile(beta5, (50, 1)).T
        return eps2

    def _kl_loss_mv(self, mu, logvar):
        
        mu_target = torch.FloatTensor([1.065, 1.068, 0.0, 0.072, 0.083])
        mu_target = mu_target.repeat(self.batch_size, 1)
        #print('MT', mu_target.shape)
        mu_target = mu_target.to('cuda').clone()

        cov_target = torch.FloatTensor([[0.265, -0.094, -0.014, -0.023, 0.124],
                                        [-0.094, 0.332, 0.0106, 0.056, -0.054],
                                        [-0.014, 0.011, 0.183, 0.0734, -0.2102],
                                        [-0.023, 0.056, 0.0734, 0.107, -0.028],
                                        [0.124, -0.054, -0.2102, -0.028, 0.564]])
        cov_target = cov_target.to('cuda').clone() #.repeat(self.batch_size, 1, 1)W
        p = torch.distributions.multivariate_normal.MultivariateNormal(mu_target, cov_target)

        #Now let's build the data distribution 
        
        mu_data = torch.ones(self.batch_size, 5)
        mu_data[:, 0] = torch.mean(mu[:, 0: 50], dim=1) #.detach()
        mu_data[:, 1] = torch.mean(mu[:, 50: 100], dim=1)
        mu_data[:, 2] = torch.mean(mu[:, 100: 150], dim=1)
        mu_data[:, 3] = torch.mean(mu[:, 150: 200], dim=1)
        mu_data[:, 4] = torch.mean(mu[:, 200: 250], dim=1)
        mu_data = mu_data.to('cuda')

        #Creating covariance matrix for data (ouput of fc_std)
        
        std_data = logvar.cpu().detach().numpy()

        x1 = np.mean(std_data[:, 0:50], axis = 0)
        x2 = np.mean(std_data[:, 50:100], axis = 0)
        x3 = np.mean(std_data[:, 100:150], axis = 0)
        x4 = np.mean(std_data[:, 150:200], axis = 0)
        x5 = np.mean(std_data[:, 200:250], axis = 0)

        X = np.array([x1, x2, x3, x4, x5]).reshape(5, -1) 
        
        cov_data = np.cov(X, rowvar=True) 
        cov_data = torch.FloatTensor(cov_data)
        cov_data = cov_data.to('cuda').clone()


        q = torch.distributions.multivariate_normal.MultivariateNormal(mu_data, cov_data) 
        kld_loss = torch.distributions.kl_divergence(q, p)
        loss = kld_loss.mean()
        return loss 