import os
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn 
from diffusers import AutoencoderKL, DDPMScheduler
from network import *
from models.GraphBrep import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim.lr_scheduler as lr_scheduler

class SurfVAETrainer():
    """ Surface VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoencoderKL(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )

        # Load pretrained surface vae (fast encode version)
        if args.finetune:
            model.load_state_dict(torch.load(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env,mode="offline")

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=8)
        return
    
        
    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for surf_uv in self.train_dataloader:
            with torch.cuda.amp.autocast():
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2)
                self.optimizer.zero_grad() # zero gradient

                # Pass through VAE 
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                # Loss functions
                kl_loss = posterior.kl().mean()
                mse_loss = loss_fn(dec, surf_uv) 
                total_loss = mse_loss + 1e-6*kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)  
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for surf_uv in self.val_dataloader:
                surf_uv = surf_uv.to(self.device).permute(0,3,1,2)
                
                posterior = self.model.encode(surf_uv).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, surf_uv).mean((1,2,3)).sum().item()
                total_loss += loss
                total_count += len(surf_uv)

        mse = total_loss/total_count
        self.model.train() # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse
    

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return


class EdgeVAETrainer():
    """ Edge VAE Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoencoderKL1D(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
            up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
            block_out_channels=[128, 256, 512],  
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512
        )

        # Load pretrained surface vae (fast encode version)
        if args.finetune:
            model.load_state_dict(torch.load(args.weight))

        self.model = model.to(self.device).train()

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4, 
            weight_decay=1e-5
        )
        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env,mode="offline")

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=8)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=8)
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()
        loss_fn = nn.MSELoss()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for edge_u in self.train_dataloader:
            with torch.cuda.amp.autocast():
                edge_u = edge_u.to(self.device).permute(0,2,1)
                self.optimizer.zero_grad() # zero gradient

               # Pass through VAE 
                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample
                
                # Loss functions
                kl_loss =  0.5 * torch.sum(
                    torch.pow(posterior.mean, 2) + posterior.var - 1.0 - posterior.logvar,
                    dim=[1, 2],
                ).mean()            
                mse_loss = loss_fn(dec, edge_u) 
                total_loss = mse_loss + 1e-6*kl_loss

                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=5.0)  
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-mse": mse_loss, "Loss-kl": kl_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_loss = 0
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        with torch.no_grad():
            for edge_u in self.val_dataloader:
                edge_u = edge_u.to(self.device).permute(0,2,1)
                
                posterior = self.model.encode(edge_u).latent_dist
                z = posterior.sample()
                dec = self.model.decode(z).sample

                loss = mse_loss(dec, edge_u).mean((1,2)).sum().item()
                total_loss += loss
                total_count += len(edge_u)

        mse = total_loss/total_count
        self.model.train() # set to train
        wandb.log({"Val-mse": mse}, step=self.iters)
        return mse
    

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        return
    

class SurfPosTrainer():
    """ Surface Position Trainer (3D bbox) """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize network
        model = SurfPosNet(self.use_cf)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.amp.GradScaler('cuda')

        # Initialize wandb
        wandb.init(project='BrepGen', dir=self.save_dir, name=args.env,mode="offline")

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=16)
        self.load_states()                                
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.amp.autocast('cuda'):
                if self.use_cf:
                    data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                    surfPos, class_label = data_cuda 
                else:
                    surfPos = data.to(self.device)
                    class_label = None

                bsz = len(surfPos)
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)

                # Predict noise
                surfPos_pred = self.model(surfPos_diffused, timesteps, class_label, True)
              
                # Compute loss
                total_loss = self.loss_fn(surfPos_pred, surfPos_noise)
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()
            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),
            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')
        total_loss = [0,0,0,0,0]

        for data in self.val_dataloader:
            if self.use_cf:
                data_cuda = [dat.to(self.device) for dat in data] # map to gpu
                surfPos, class_label = data_cuda 
            else:
                surfPos = data.to(self.device)
                class_label = None

            bsz = len(surfPos)
        
            total_count += len(surfPos)
            
            for idx, step in enumerate([10,50,100,200,500]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                surfPos_noise = torch.randn(surfPos.shape).to(self.device)  
                surfPos_diffused = self.noise_scheduler.add_noise(surfPos, surfPos_noise, timesteps)
                with torch.no_grad():
                    surfPos_pred = self.model(surfPos_diffused, timesteps, class_label) 
                loss = mse_loss(surfPos_pred, surfPos_noise).mean((1,2)).sum().item()
                total_loss[idx] += loss

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]}, step=self.iters)
        return
    

    def load_states(self):
        ckpt_dir=os.path.join(self.save_dir,'last.pth')
        if not os.path.exists(ckpt_dir):
            print("no checkpoint,train from scratch")
            pass
        else:
            loaded_state = torch.load(ckpt_dir, weights_only=True,map_location=self.device)
            self.optimizer.load_state_dict(loaded_state['optimizer'])
            #self.model.load_state_dict(loaded_state['model'], strict=False)
            self.scaler.load_state_dict(loaded_state['scaler'])
            self.epoch = loaded_state['epoch']
            self.iters = loaded_state['iters']
            new_state_dict = {}
            for key, value in loaded_state['model'].items():
                new_key = key
                if not key.startswith('module.'):
                    new_key = 'module.' + key
                new_state_dict[new_key] = value
                
            self.model.load_state_dict(new_state_dict)
            


            print(f"Get checkpoint,train from epoch{self.epoch}")
        return
    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),
            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        return



class SurfZTrainer():
    """ Surface Latent Geometry Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize network
        model = SurfZNet(self.use_cf)
        model = nn.DataParallel(model) # distributed training 
        self.model = model.to(self.device).train()

        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        surf_vae.load_state_dict(torch.load(args.surfvae,weights_only=True), strict=False)
        surf_vae = nn.DataParallel(surf_vae) # distributed inference 
        self.surf_vae = surf_vae.to(self.device).eval()

        self.loss_fn = nn.MSELoss()

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )

        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        # Initialize wandb
        wandb.init(project='BrepGen', dir=args.save_dir, name=args.env,mode="offline")

        # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=16)
        self.load_states()
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.amp.autocast('cuda'):
                data_cuda = [x.to(self.device) for x in data] # map to gpu               
                if self.use_cf:
                    surfPos, surfPnt, surf_mask, class_label = data_cuda 
                else:
                    surfPos, surfPnt, surf_mask = data_cuda
                    class_label = None

                bsz = len(surfPos)

                # Augment the surface position (see https://arxiv.org/abs/2106.15282)
                conditions = [surfPos]
                aug_data = []
                for data in conditions:
                    aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(data.shape).to(self.device)  
                    aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                surfPos = aug_data[0]

                # Pass through surface VAE to sample latent z 
                # with torch.no_grad():
                #     surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                #     surf_z = self.surf_vae(surf_uv)
                #     surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)     
                with torch.no_grad():
                    sub_batch_size = 16000 
                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z_list = []
                    for i in range(0, surf_uv.size(0), sub_batch_size):
                        sub_batch_uv = surf_uv[i:i+sub_batch_size]
                        sub_batch_z = self.surf_vae(sub_batch_uv)
                        surf_z_list.append(sub_batch_z)
                    surf_z = torch.cat(surf_z_list, dim=0)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)

                surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z
                
                self.optimizer.zero_grad() # zero gradient

                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                surfZ_noise = torch.randn(surfZ.shape).to(self.device)  
                surfZ_diffused = self.noise_scheduler.add_noise(surfZ, surfZ_noise, timesteps)

                # Predict noise
                surfZ_pred = self.model(surfZ_diffused, timesteps, surfPos, surf_mask, class_label, True)

                # Loss
                total_loss = self.loss_fn(surfZ_pred[~surf_mask], surfZ_noise[~surf_mask])        
             
                # Update model
                self.scaler.scale(total_loss).backward()
                nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # logging
            if self.iters % 10 == 0:
                wandb.log({"Loss-noise": total_loss}, step=self.iters)

            self.iters += 1
            progress_bar.update(1)

        progress_bar.close()
        self.epoch += 1 
        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),
            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        return 
    

    def test_val(self):
        """
        Test the model on validation set
        """
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0]*5

        for data in self.val_dataloader:
            data_cuda = [x.to(self.device) for x in data] # map to gpu
            if self.use_cf:
                surfPos, surfPnt, surf_mask, class_label = data_cuda 
            else:
                surfPos, surfPnt, surf_mask = data_cuda
                class_label = None
            bsz = len(surfPos)

            # Pass through surface VAE to sample latent z 
            # with torch.no_grad():
            #     surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
            #     surf_z = self.surf_vae(surf_uv)
            # surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)    
            with torch.no_grad():
                sub_batch_size = 16000 
                surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                surf_z_list = []
                for i in range(0, surf_uv.size(0), sub_batch_size):
                    sub_batch_uv = surf_uv[i:i+sub_batch_size]
                    sub_batch_z = self.surf_vae(sub_batch_uv)
                    surf_z_list.append(sub_batch_z)
                surf_z = torch.cat(surf_z_list, dim=0)
                surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2) 
            tokens = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

            total_count += len(surfPos)
            
            for idx, step in enumerate([10,50,100,200,500]):
                # Evaluate at timestep 
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long()  # [batch,]
                noise = torch.randn(tokens.shape).to(self.device)  
                diffused = self.noise_scheduler.add_noise(tokens, noise, timesteps)
                with torch.no_grad():
                    pred = self.model(diffused, timesteps, surfPos, surf_mask, class_label)
                loss = mse_loss(pred[~surf_mask], noise[~surf_mask]).mean(-1).sum().item()
                total_loss[idx] += loss

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss/total_count for loss in total_loss]
        self.model.train() # set to train
        wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2], "Val-200": mse[3], "Val-500": mse[4]}, step=self.iters)
        return
    

    def load_states(self):
        ckpt_dir=os.path.join(self.save_dir,'last.pth')
        if not os.path.exists(ckpt_dir):
            print("no checkpoint,train from scratch")
            pass
        else:
            loaded_state = torch.load(ckpt_dir, weights_only=True,map_location=self.device)
            self.optimizer.load_state_dict(loaded_state['optimizer'])
            #self.model.load_state_dict(loaded_state['model'], strict=False)
            self.scaler.load_state_dict(loaded_state['scaler'])
            self.epoch = loaded_state['epoch']
            self.iters = loaded_state['iters']
            new_state_dict = {}
            for key, value in loaded_state['model'].items():
                new_key = key
                if not key.startswith('module.'):
                    new_key = 'module.' + key
                new_state_dict[new_key] = value
                
            self.model.load_state_dict(new_state_dict)
            


            print(f"Get checkpoint,train from epoch{self.epoch}")
        return
    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),
            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        return



class GEdgePosTrainer():
    """ Edge Latent Z Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.max_edge = args.max_edge
        self.max_face = args.max_face
        self.num_scale = 1
        # if args.one_edge:
        #     self.num_scale = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize network
        #model = GEdgePosNet_ATT(self.use_cf)
        #model = EdgePosNet_refine(self.use_cf)
        #model = EdgePosNet_trans(self.use_cf)
        model = CGTD_EP(self.use_cf)
        #model = EdgePosNet_cbranch(self.use_cf)
        
        #model = GEdgePosNet_AGL_noC(self.use_cf)
        #model = GEdgePosNet_noC(self.use_cf)
        #model = GEdgePosNet_GNN_noC(self.use_cf)
        #model = GEdgePosNet_ATT_noC(self.use_cf)
        #model = GEvpACNet(self.use_cf)
        # if args.checkpoint !='':
        #     model.load_state_dict(torch.load(args.checkpoint))
        model = nn.DataParallel(model) # distributed training 
        
        self.model = model.to(self.device).train()
        
        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        surf_vae.load_state_dict(torch.load(args.surfvae,weights_only=True), strict=False)
        surf_vae = nn.DataParallel(surf_vae) # distributed inference 
        self.surf_vae = surf_vae.to(self.device).eval()

        # Load pretrained edge vae (fast encode version)
        # edge_vae = AutoencoderKL1DFastEncode(
        #     in_channels=3,
        #     out_channels=3,
        #     down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        #     up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        #     block_out_channels=[128, 256, 512],  
        #     layers_per_block=2,
        #     act_fn='silu',
        #     latent_channels=3,
        #     norm_num_groups=32,
        #     sample_size=512
        # )
        # edge_vae.load_state_dict(torch.load(args.edgevae), strict=False)
        # edge_vae = nn.DataParallel(edge_vae) # distributed inference 
        # self.edge_vae = edge_vae.to(self.device).eval()

        self.loss_fn = nn.MSELoss(reduction='none')

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )


        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
            # lr=5e-4,
            # betas=(0.95, 0.999),
            # weight_decay=1e-6,
            # eps=1e-08,
                # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=16)
        self.scaler = torch.amp.GradScaler('cuda')
        self.load_states()
        #print(args.bbox_scaled)
        #self.lr_scheduler_func = get_lr_scheduler(5e-5, 5e-7, 1000)
        # Initialize wandb

        # wandb.init(project='BrepGen', dir=args.save_dir, name=args.env,mode="offline")

       
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            #torch.cuda.empty_cache()
            with torch.amp.autocast('cuda'):
                data_cuda = [x.to(self.device) for x in data] # map to gpu
                if self.use_cf:
                    edgePos,surfPnt, surfPos, _,face_mask,edge_mask, class_label = data_cuda
                else:
                    edgePos,surfPnt, surfPos, _,face_mask,edge_mask = data_cuda
                    class_label = None

                bsz = len(surfPos)

                # Pass through surface/edge VAE to sample latent z 
                with torch.no_grad():
                    # sub_batch_size = bsz* self.max_face
                    # surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    # surf_z_list = []
                    # for i in range(0, surf_uv.size(0), sub_batch_size):
                    #     sub_batch_uv = surf_uv[i:i+sub_batch_size]
                    #     sub_batch_z = self.surf_vae(sub_batch_uv)
                    #     surf_z_list.append(sub_batch_z)
                    # surf_z = torch.cat(surf_z_list, dim=0)
                    # surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)

                    # surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)  
                    surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

                    edge_mask=edge_mask.unsqueeze(-1)

                    

                # Augment the surface position and latent (see https://arxiv.org/abs/2106.15282)
                conditions = [surfPos, surfZ]
                aug_data = []
                for data in conditions:
                    aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(data.shape).to(self.device)
                    aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                surfPos, surfZ = aug_data[0], aug_data[1]
                #c_evp=compute_boxx(surfPos)*edge_mask
                #edge_mask=c_evp[...,0:1]
                # Zero gradient
                self.optimizer.zero_grad() 

                
                #print(self.noise_scheduler.config.num_train_timesteps)
                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                #timesteps = torch.randint(0, 200, (bsz,), device=self.device).long()  # [batch,]


                edgePos_noise = torch.randn((bsz,6,self.max_face,self.max_face)).to(self.device) 
                edgePos_noise = (torch.tril(edgePos_noise, -1) + torch.tril(edgePos_noise, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask
                edgePos_diffused = self.noise_scheduler.add_noise(edgePos, edgePos_noise, timesteps)*edge_mask
                
                #assert (edgePos_diffused==edgePos_diffused.transpose(1, 2)).all()
                # if self.use_cf:
                #     edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label, True)
                # else:
                edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,face_mask,edge_mask)
                #edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label, True)
                #edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,edge_mask,class_label, True)
                # batch*maxface*54          joint_face_diffused
                # batch*24*maxface*maxface  sparse_edgef_diffused
                # batch*1*maxface*maxface   standard_adj_diffused
                # batch                     timesteps
                # batch*maxface             face_mask
                # batch*1*maxface*maxface   edge_mask
                # batch*1                   class_label
                total_loss=self.loss_fn(edgePos_pred, edgePos_noise)

                adj_mask = edge_mask.expand(-1, -1, -1, 6)  # [batch, len, len, 6]

                # 计算有效部分的损失
                masked_loss = total_loss * adj_mask  # [batch, len, len, 6]

                # 计算总损失（仅考虑有效部分）
                batch_loss = masked_loss.sum(dim=[1, 2, 3]) / adj_mask.sum(dim=[1, 2, 3]).clamp(min=1)
                
                #total_loss=torch.sum((total_loss * edge_mask).reshape(bsz, -1), dim=-1)/ (torch.sum(edge_mask.reshape(bsz, -1), dim=-1)*12) #/12
                total_loss = batch_loss.mean()
                
                current_lr = self.optimizer.param_groups[0]['lr']


                if self.iters<2000:
                    back_loss = total_loss*((self.iters+1)/2000)
                else:
                    back_loss = total_loss
                # Update model
            self.scaler.scale(back_loss).backward()
            nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.iters += 1
            progress_bar.update(1)
            progress_bar.set_postfix(total=f"{total_loss.item():.4f}",
                                     lr=f"{10000*current_lr:.3f}") #输入一个字典，显示实验指标

        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),
            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        progress_bar.close()
        self.epoch += 1 
        return 
    

  
    def test_val(self):
        """
        Test the model on validation set
        """
        #torch.cuda.empty_cache()
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0]*3

        for data in self.val_dataloader:
            data_cuda = [x.to(self.device) for x in data] # map to gpu
            if self.use_cf:
                edgePos,surfPnt, surfPos, _,face_mask,edge_mask, class_label = data_cuda
            else:
                edgePos,surfPnt, surfPos, _,face_mask,edge_mask = data_cuda
                class_label = None
            bsz = len(surfPos)

            # Pass through surface/edge VAE to sample latent z 
            
            # sub_batch_size = 6400 
            # surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
            # surf_z_list = []
            # for i in range(0, surf_uv.size(0), sub_batch_size):
            #     sub_batch_uv = surf_uv[i:i+sub_batch_size]
            #     sub_batch_z = self.surf_vae(sub_batch_uv)
            #     surf_z_list.append(sub_batch_z)
            # surf_z = torch.cat(surf_z_list, dim=0)
            # surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)
            # surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

            surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
            surf_z = self.surf_vae(surf_uv)
            surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)  
            surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z


            edge_mask=edge_mask.unsqueeze(-1)

            total_count += len(surfPos)
            

            for idx, step in enumerate([10,50,100]):
                # Evaluate at timestep   
                # Add noise
                #timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                
                timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long() 
                edgePos_noise = torch.randn((bsz,6,self.max_face,self.max_face)).to(self.device) 
                edgePos_noise = (torch.tril(edgePos_noise, -1) + torch.tril(edgePos_noise, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask
                edgePos_diffused = self.noise_scheduler.add_noise(edgePos, edgePos_noise, timesteps)*edge_mask
                #assert (edgePos_diffused==edgePos_diffused.transpose(1, 2)).all()
                # if self.use_cf:
                #     edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label)
                # else:
                edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,face_mask,edge_mask)
                #edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label)
                #edgePos_pred = self.model(edgePos_diffused,surfPos, surfZ,timesteps,edge_mask,class_label)
                # batch*maxface*54          joint_face_diffused 
                # batch*24*maxface*maxface  sparse_edgef_diffused
                # batch*1*maxface*maxface   standard_adj_diffused
                # batch                     timesteps
                # batch*maxface             face_mask
                # batch*1*maxface*maxface   edge_mask
                # batch*1                   class_label
                loss=self.loss_fn(edgePos_pred, edgePos_noise)
                adj_mask = edge_mask.expand(-1, -1, -1, 6)  # [batch, len, len, 6]
                # 计算有效部分的损失
                masked_loss = loss * adj_mask  # [batch, len, len, 6]
                # 计算总损失（仅考虑有效部分）
                batch_loss = masked_loss.sum(dim=[1, 2, 3]) / adj_mask.sum(dim=[1, 2, 3]).clamp(min=1)
                total_loss[idx] += batch_loss.sum().item()
                

            progress_bar.update(1)
        progress_bar.close()

        mse = [loss/total_count for loss in total_loss]
        #torch.cuda.empty_cache()
        self.model.train() # set to train
        # wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2]}, step=self.iters)
        return
    
    
    def load_states(self):
        ckpt_dir=os.path.join(self.save_dir,'last.pth')
        if not os.path.exists(ckpt_dir):
            print("no checkpoint,train from scratch")
            pass
        else:
            loaded_state = torch.load(ckpt_dir, map_location=self.device,weights_only=True)
            self.optimizer.load_state_dict(loaded_state['optimizer'])
            self.scaler.load_state_dict(loaded_state['scaler'])
            self.epoch = loaded_state['epoch']
            self.iters = loaded_state['iters']
            new_state_dict = {}
            for key, value in loaded_state['model'].items():
                new_key = key
                if not key.startswith('module.'):
                    new_key = 'module.' + key
                new_state_dict[new_key] = value
                
            self.model.load_state_dict(new_state_dict,strict=False)
            


            print(f"Get checkpoint,train from epoch{self.epoch}")
        return

    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),
            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        return




class GEdgeZTrainer():
    """ Edge Latent Z Trainer """
    def __init__(self, args, train_dataset, val_dataset): 
        # Initilize model and load to gpu
        self.iters = 0
        self.epoch = 0
        self.save_dir = args.save_dir
        self.use_cf = args.cf
        self.z_scaled = args.z_scaled
        self.max_edge = args.max_edge
        self.max_face = args.max_face
        self.num_scale = 1
        # if args.one_edge:
        #     self.num_scale = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize network
        #model = GEdgeZNet(self.use_cf)
        #model = GEdgeZNet_noC(self.use_cf)
        #model = GEdgeZNet_GNN_noC(self.use_cf)
        #model = GEdgeZNet_AGL_noC(self.use_cf)
        #model = EdgeZNet_refine(self.use_cf)
        #model = EdgeZNet_trans(self.use_cf)
        model = CGTD_EZ(self.use_cf)
        #model = EdgeZNet_cbranch(self.use_cf)
        
        #model = EdgeZNet_ori(self.use_cf)
        
        # if args.checkpoint !='':
        #     model.load_state_dict(torch.load(args.checkpoint))
        model = nn.DataParallel(model) # distributed training 
        
        self.model = model.to(self.device).train()
        
        # Load pretrained surface vae (fast encode version)
        surf_vae = AutoencoderKLFastEncode(in_channels=3,
            out_channels=3,
            down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
            up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
            block_out_channels=[128, 256, 512, 512],
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512,
        )
        surf_vae.load_state_dict(torch.load(args.surfvae,weights_only=True), strict=False)
        surf_vae = nn.DataParallel(surf_vae) # distributed inference 
        self.surf_vae = surf_vae.to(self.device).eval()

        # Load pretrained edge vae (fast encode version)
        edge_vae = AutoencoderKL1DFastEncode(
            in_channels=3,
            out_channels=3,
            down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
            up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
            block_out_channels=[128, 256, 512],  
            layers_per_block=2,
            act_fn='silu',
            latent_channels=3,
            norm_num_groups=32,
            sample_size=512
        )
        edge_vae.load_state_dict(torch.load(args.edgevae,weights_only=True), strict=False)
        edge_vae = nn.DataParallel(edge_vae) # distributed inference 
        self.edge_vae = edge_vae.to(self.device).eval()

        self.loss_fn = nn.MSELoss(reduction='none')

        # Initialize diffusion scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule='linear',
            prediction_type='epsilon',
            beta_start=0.0001,
            beta_end=0.02,
            clip_sample=False,
        )


        # Initialize optimizer
        self.network_params = list(self.model.parameters())
        
        self.optimizer = torch.optim.AdamW(
            self.network_params,
            lr=5e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
            eps=1e-08,
        )
            # lr=5e-4,
            # betas=(0.95, 0.999),
            # weight_decay=1e-6,
            # eps=1e-08,
                # Initilizer dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=True, 
                                                batch_size=args.batch_size,
                                                num_workers=16)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=False, 
                                             batch_size=args.batch_size,
                                             num_workers=16)
        self.scaler = torch.amp.GradScaler('cuda')
        self.load_states()
        
        #self.lr_scheduler_func = get_lr_scheduler(5e-5, 5e-7, 1000)
        # Initialize wandb

        # wandb.init(project='BrepGen', dir=args.save_dir, name=args.env,mode="offline")

       
        return
    

    def train_one_epoch(self):
        """
        Train the model for one epoch
        """
        self.model.train()

        progress_bar = tqdm(total=len(self.train_dataloader))
        progress_bar.set_description(f"Epoch {self.epoch}")

        # Train    
        for data in self.train_dataloader:
            with torch.amp.autocast('cuda'):
                #torch.cuda.empty_cache()
                data_cuda = [x.to(self.device) for x in data] # map to gpu
                if self.use_cf:
                    evp,edgePnt,surfPnt, surfPos, adj,face_mask,edge_mask, class_label = data_cuda
                else:
                    evp,edgePnt,surfPnt, surfPos, adj,face_mask,edge_mask = data_cuda
                    class_label = None

                bsz = len(surfPos)

                # Pass through surface/edge VAE to sample latent z 
                with torch.no_grad():
                    #adj= adj>0
                    # sub_batch_size = 6400 
                    # surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    # surf_z_list = []
                    # for i in range(0, surf_uv.size(0), sub_batch_size):
                    #     sub_batch_uv = surf_uv[i:i+sub_batch_size]
                    #     sub_batch_z = self.surf_vae(sub_batch_uv)
                    #     surf_z_list.append(sub_batch_z)
                    # surf_z = torch.cat(surf_z_list, dim=0)
                    # surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)
                    # surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

                    surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                    surf_z = self.surf_vae(surf_uv)
                    surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)  
                    surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z
                    #assert (adj==adj.transpose(1, 2)).all()

                    adj = adj>0
                    # 获取需要处理的 edge 的索引
                    adj_flatten = adj.flatten(0,1).flatten(0,1).squeeze(-1)  # 将 adj 展平 (batch*n*n, 1) -> (batch*n*n)

                    # 筛选出需要经过 VAE 的边
                    edge_u = edgePnt.flatten(0,1).flatten(0,1).permute(0,2,1)
                    edge_u_selected = edge_u[adj_flatten]  # 只选出需要处理的边

                    # 处理选出的边
                    sub_batch_size = bsz * self.max_face * self.max_face //2
                    edge_z_list = []
                    for i in range(0, edge_u_selected.size(0), sub_batch_size):
                        sub_batch_u = edge_u_selected[i:i+sub_batch_size]
                        sub_batch_z = self.edge_vae(sub_batch_u)
                        edge_z_list.append(sub_batch_z)

                    # 将结果拼接起来
                    edge_z_selected = torch.cat(edge_z_list, dim=0).permute(0,2,1).flatten(-2,-1)

                    # 初始化 edgeZ 全零矩阵，尺寸为 (batch, n, n, 12)
                    edgeZ = torch.zeros((bsz, self.max_face, self.max_face, 12), device=edge_z_selected.device).to(edge_z_selected.dtype)

                    # 将处理后的结果重新填充到 edgeZ 中
                    edgeZ_flatten = edgeZ.flatten(0,1).flatten(0,1)  # (batch*n*n, 12)
                    edgeZ_flatten[adj_flatten] = edge_z_selected  # 只填充选中的边

                    # 恢复 edgeZ 的原始维度
                    edgeZ = edgeZ_flatten.unflatten(0, (-1, self.max_face)).unflatten(0, (bsz,-1))* self.z_scaled
                    #tolerance = 1e-3  # 设置容忍的误差范围，例如 0.001
                    #assert torch.allclose(edgeZ, edgeZ.transpose(1, 2), atol=1e-2), "edgeZ 和其转置在某些元素上存在差异"
                    edgeZ=(edgeZ+edgeZ.transpose(1, 2))/2
                    #assert (edgeZ==edgeZ.transpose(1, 2)).all()
                    # edge_u = edgePnt.flatten(0,1).flatten(0,1).permute(0,2,1)
                    # sub_batch_size=sub_batch_size*self.max_face // 2   
                    # edge_z_list = []
                    # for i in range(0, edge_u.size(0), sub_batch_size):
                    #     sub_batch_u = edge_u[i:i+sub_batch_size]
                    #     sub_batch_z = self.edge_vae(sub_batch_u)
                    #     edge_z_list.append(sub_batch_z)
                    # edge_z = torch.cat(edge_z_list, dim=0)
                    # edge_z = edge_z.unflatten(0, (-1, self.max_face)).unflatten(0, (bsz,-1)).permute(0,1,2,4,3)

                    edge_mask=adj.float().unsqueeze(-1)
                    #edgeZ = edge_z.flatten(-2,-1)  * self.z_scaled
                    #assert (adj==adj.transpose(1, 2)).all()
                    #adj=adj.unsqueeze(-1)
                edgePos=evp[...,:6]
                edgeZV=torch.cat([edgeZ,evp[...,-6:]],dim=-1)
                #edge_mask=edge_mask.unsqueeze(-1)
                

                #standard_adj=torch.concat([standard_adj,sparse_edgef[:,12:]],dim=1)

                # Augment the surface position and latent (see https://arxiv.org/abs/2106.15282)
                conditions = [edgePos,surfPos, surfZ]
                aug_data = []
                for data in conditions:
                    aug_timesteps = torch.randint(0, 15, (bsz,), device=self.device).long()
                    aug_noise = torch.randn(data.shape).to(self.device)
                    aug_data.append(self.noise_scheduler.add_noise(data, aug_noise, aug_timesteps))
                edgePos,surfPos, surfZ = aug_data[0], aug_data[1],aug_data[2]
                edgePos=(edgePos+edgePos.transpose(1,2))/2
                # Zero gradient
                self.optimizer.zero_grad() 

                
                
                # Add noise
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]

                edgeZV_noise = torch.randn((bsz,18,self.max_face,self.max_face)).to(self.device) 
                edgeZV_noise = (torch.tril(edgeZV_noise, -1) + torch.tril(edgeZV_noise, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask
                edgeZV_diffused = self.noise_scheduler.add_noise(edgeZV, edgeZV_noise, timesteps)*edge_mask
                #assert (edgeZV_diffused==edgeZV_diffused.transpose(1, 2)).all()
                # if self.use_cf:
                #     edgeZV_pred = self.model(edgeZV_diffused,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label, True)
                # else:
                edgeZV_pred = self.model(edgeZV_diffused,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
            
                
                #edgeZV_pred = self.model(edgeZV_diffused,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label, True)
                
                # batch*maxface*54          joint_face_diffused 
                # batch*24*maxface*maxface  sparse_edgef_diffused
                # batch*1*maxface*maxface   standard_adj_diffused
                # batch                     timesteps
                # batch*maxface             face_mask
                # batch*1*maxface*maxface   edge_mask
                # batch*1                   class_label

                total_loss=self.loss_fn(edgeZV_pred, edgeZV_noise)
                
                adj_mask = edge_mask.expand(-1, -1, -1, 18)  # [batch, len, len, 18]

                # 计算有效部分的损失
                masked_loss = total_loss * adj_mask  # [batch, len, len, 18]

                # 计算总损失（仅考虑有效部分）
                batch_loss = masked_loss.sum(dim=[1, 2, 3]) / adj_mask.sum(dim=[1, 2, 3]).clamp(min=1)
                
                #total_loss=torch.sum((total_loss * edge_mask).reshape(bsz, -1), dim=-1)/ (torch.sum(edge_mask.reshape(bsz, -1), dim=-1)*12) #/12
                total_loss = batch_loss.mean()
                
                current_lr = self.optimizer.param_groups[0]['lr']


                if self.iters<2000:
                    back_loss = total_loss*((self.iters+1)/2000)
                else:
                    back_loss = total_loss
                # Update model
            self.scaler.scale(back_loss).backward()
            nn.utils.clip_grad_norm_(self.network_params, max_norm=50.0) # clip gradient
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.iters += 1
            progress_bar.update(1)
            progress_bar.set_postfix(total=f"{total_loss.item():.4f}",
                                     lr=f"{10000*current_lr:.3f}") #输入一个字典，显示实验指标

        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),

            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        progress_bar.close()
        self.epoch += 1 
        return 
    

  
    def test_val(self):
        """
        Test the model on validation set
        """
        #torch.cuda.empty_cache()
        self.model.eval() # set to eval
        total_count = 0
        mse_loss = nn.MSELoss(reduction='none')

        progress_bar = tqdm(total=len(self.val_dataloader))
        progress_bar.set_description(f"Testing")

        total_loss = [0]*3

        for data in self.val_dataloader:
            data_cuda = [x.to(self.device) for x in data] # map to gpu
            if self.use_cf:
                evp,edgePnt,surfPnt, surfPos, adj,face_mask,edge_mask, class_label = data_cuda
            else:
                evp,edgePnt,surfPnt, surfPos, adj,face_mask,edge_mask = data_cuda
                class_label = None

            bsz = len(surfPos)

            # Pass through surface/edge VAE to sample latent z 
            with torch.no_grad():
                # sub_batch_size = 6400 
                # surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                # surf_z_list = []
                # for i in range(0, surf_uv.size(0), sub_batch_size):
                #     sub_batch_uv = surf_uv[i:i+sub_batch_size]
                #     sub_batch_z = self.surf_vae(sub_batch_uv)
                #     surf_z_list.append(sub_batch_z)
                # surf_z = torch.cat(surf_z_list, dim=0)
                # surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)
                # surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z


                surf_uv = surfPnt.flatten(0,1).permute(0,3,1,2)
                surf_z = self.surf_vae(surf_uv)
                surf_z = surf_z.unflatten(0, (bsz, -1)).flatten(-2,-1).permute(0,1,3,2)  
                surfZ = surf_z.flatten(-2,-1)  * self.z_scaled # rescaled the latent z

                adj = adj>0
                adj_flatten = adj.flatten(0,1).flatten(0,1).squeeze(-1).bool()  # 将 adj 展平 (batch*n*n, 1) -> (batch*n*n)

                # 筛选出需要经过 VAE 的边
                edge_u = edgePnt.flatten(0,1).flatten(0,1).permute(0,2,1)
                edge_u_selected = edge_u[adj_flatten]  # 只选出需要处理的边

                # 处理选出的边
                sub_batch_size = bsz * self.max_face * self.max_face // 2
                edge_z_list = []
                for i in range(0, edge_u_selected.size(0), sub_batch_size):
                    sub_batch_u = edge_u_selected[i:i+sub_batch_size]
                    sub_batch_z = self.edge_vae(sub_batch_u)
                    edge_z_list.append(sub_batch_z)

                # 将结果拼接起来
                edge_z_selected = torch.cat(edge_z_list, dim=0).permute(0,2,1).flatten(-2,-1)

                # 初始化 edgeZ 全零矩阵，尺寸为 (batch, n, n, 12)
                edgeZ = torch.zeros((bsz, self.max_face, self.max_face, 12), device=edge_z_selected.device).to(edge_z_selected.dtype)

                # 将处理后的结果重新填充到 edgeZ 中
                edgeZ_flatten = edgeZ.flatten(0,1).flatten(0,1)  # (batch*n*n, 12)
                edgeZ_flatten[adj_flatten] = edge_z_selected  # 只填充选中的边

                # 恢复 edgeZ 的原始维度
                edgeZ = edgeZ_flatten.unflatten(0, (-1, self.max_face)).unflatten(0, (bsz,-1))* self.z_scaled
                #tolerance = 1e-3  # 设置容忍的误差范围，例如 0.001
                #assert torch.allclose(edgeZ, edgeZ.transpose(1, 2), atol=1e-2), "edgeZ 和其转置在某些元素上存在差异"
                edgeZ=(edgeZ+edgeZ.transpose(1, 2))/2
                edge_mask=adj.float().unsqueeze(-1)
            edgePos=evp[...,:6]
            edgeZV=torch.cat([edgeZ,evp[...,-6:]],dim=-1)
            

            #joint_data = torch.concat([edgeZ, vertPos], -1) # vertex as part of edge, 18D total

            total_count += len(surfPos)
            

            for idx, step in enumerate([10,50,100]):
                # Evaluate at timestep 
                with torch.no_grad():    
                    # Add noise
                    #timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()  # [batch,]
                    
                    timesteps = torch.randint(step-1, step, (bsz,), device=self.device).long() 
                    edgeZV_noise = torch.randn((bsz,18,self.max_face,self.max_face)).to(self.device) 
                    edgeZV_noise = (torch.tril(edgeZV_noise, -1) + torch.tril(edgeZV_noise, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask
                    edgeZV_diffused = self.noise_scheduler.add_noise(edgeZV, edgeZV_noise, timesteps)*edge_mask
                    #assert (edgeZV_diffused==edgeZV_diffused.transpose(1, 2)).all()
                    # if self.use_cf:
                    #     edgeZV_pred = self.model(edgeZV_diffused,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label)
                    # else:
                    edgeZV_pred = self.model(edgeZV_diffused,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                    #edgeZV_pred = self.model(edgeZV_diffused,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask,class_label)

                    # batch*maxface*54          joint_face_diffused 
                    # batch*24*maxface*maxface  sparse_edgef_diffused
                    # batch*1*maxface*maxface   standard_adj_diffused
                    # batch                     timesteps
                    # batch*maxface             face_mask
                    # batch*1*maxface*maxface   edge_mask
                    # batch*1                   class_label
                    loss=self.loss_fn(edgeZV_pred, edgeZV_noise)
                    adj_mask = edge_mask.expand(-1, -1, -1, 18)  # [batch, len, len, 18]
                    # 计算有效部分的损失
                    masked_loss = loss * adj_mask  # [batch, len, len, 18]
                    # 计算总损失（仅考虑有效部分）
                    batch_loss = masked_loss.sum(dim=[1, 2, 3]) / adj_mask.sum(dim=[1, 2, 3]).clamp(min=1)
                    total_loss[idx] += batch_loss.sum().item()
            progress_bar.update(1)
        progress_bar.close()

        mse = [loss/total_count for loss in total_loss]
        #torch.cuda.empty_cache()
        self.model.train() # set to train
        # wandb.log({"Val-010": mse[0], "Val-050": mse[1], "Val-100": mse[2]}, step=self.iters)
        return
    
    
    def load_states(self):
        ckpt_dir=os.path.join(self.save_dir,'last.pth')
        if not os.path.exists(ckpt_dir):
            print("no checkpoint,train from scratch")
            pass
        else:
            loaded_state = torch.load(ckpt_dir, weights_only=True,map_location=self.device)
            self.optimizer.load_state_dict(loaded_state['optimizer'])

            self.scaler.load_state_dict(loaded_state['scaler'])
            self.epoch = loaded_state['epoch']
            self.iters = loaded_state['iters']
            new_state_dict = {}
            for key, value in loaded_state['model'].items():
                new_key = key
                if not key.startswith('module.'):
                    new_key = 'module.' + key
                new_state_dict[new_key] = value
                
            self.model.load_state_dict(new_state_dict)
            


            print(f"Get checkpoint,train from epoch{self.epoch}")
        return
    def save_model(self):
        torch.save(self.model.module.state_dict(), os.path.join(self.save_dir,'epoch_'+str(self.epoch)+'.pt'))
        saved_state = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.model.module.state_dict(),
            'scaler':self.scaler.state_dict(),
            'epoch': self.epoch,
            'iters':self.iters
        }
        torch.save(saved_state, os.path.join(self.save_dir,'last'+'.pth'))
        return
   

   
