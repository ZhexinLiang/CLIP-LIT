from math import sqrt
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torchvision
import torch.optim
import argparse

import dataloader_prompt_margin
import dataloader_prompt_add
import dataloader_images as dataloader_sharp 

import model_small

import numpy as np

from test_function import inference

import clip_score
import random
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import clip

import pyiqa
import shutil

task_name="train0"
writer = SummaryWriter('./'+task_name+"/"+'tensorboard_'+task_name)

dstpath="./"+task_name+"/"+"train_scripts"
if not os.path.exists(dstpath):
    os.makedirs(dstpath)
shutil.copy("train.py",dstpath)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
#load clip
model, preprocess = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")#ViT-B/32
model.to(device)
for para in model.parameters():
    para.requires_grad = False

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class Prompts(nn.Module):
    def __init__(self,initials=None):
        super(Prompts,self).__init__()
        print("The initial prompts are:",initials)
        self.text_encoder = TextEncoder(model)
        if isinstance(initials,list):
            text = clip.tokenize(initials).cuda()
            # print(text)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*config.length_prompt)," ".join(["X"]*config.length_prompt)]).requires_grad_())).cuda()

    def forward(self,tensor,flag=1):
        tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*config.length_prompt)]])
        text_features = self.text_encoder(self.embedding_prompt,tokenized_prompts)
        
        for i in range(tensor.shape[0]):
            image_features=tensor[i]
            nor=torch.norm(text_features,dim=-1, keepdim=True)
            if flag==0:
                similarity = (100.0 * image_features @ (text_features/nor).T)#.softmax(dim=-1)
                if(i==0):
                    probs=similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features/nor).T).softmax(dim=-1)#/nor
                if(i==0):
                    probs=similarity[:,0]
                else:
                    probs=torch.cat([probs,similarity[:,0]],dim=0)
        return probs

def weights_init(m):
    classname = m.__class__.__name__ 
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def random_crop(img):
    b,c,h,w=img.shape
    hs=random.randint(0,h-224)
    hw=random.randint(0,w-224)
    return img[:,:,hs:hs+224,hw:hw+224]

def train(config):
    
    #load model
    U_net=model_small.UNet_emb_oneBranch_symmetry_noreflect(3,1).cuda()
  
    iqa_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
    
    #add pretrained model weights
    if config.load_pretrain_prompt == True:
        learn_prompt=Prompts(config.prompt_pretrain_dir).cuda()
        torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "pretrained_prompt" + '.pth')
    else:
        if config.num_clip_pretrained_iters < 3000:
            print("WARNING: For training from scratch, num_clip_pretrained_iters should not lower than 3000 iterations!\nAutomatically reset num_clip_pretrained_iters to 8000 iterations...")
            config.num_clip_pretrained_iters=8000
        learn_prompt=Prompts([" ".join(["X"]*(config.length_prompt))," ".join(["X"]*(config.length_prompt))]).cuda()
    learn_prompt =  torch.nn.DataParallel(learn_prompt)
    U_net.apply(weights_init)
    
    if config.load_pretrain == True:
        print("The load_pretrain is True, thus num_reconstruction_iters is automatically set to 0.")
        config.num_reconstruction_iters=0
        state_dict = torch.load(config.pretrain_dir)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        U_net.load_state_dict(new_state_dict)
        #U_net.load_state_dict(torch.load(config.pretrain_dir))
        torch.save(U_net.state_dict(), config.train_snapshots_folder + "pretrained_network" + '.pth')
    else:
        if config.num_reconstruction_iters<200:
            print("WARNING: For training from scratch, num_reconstruction_iters should not lower than 200 iterations!\nAutomatically reset num_reconstruction_iters to 1000 iterations...")
            config.num_reconstruction_iters=1000
    U_net= torch.nn.DataParallel(U_net)
    
    #load dataset
    train_dataset = dataloader_sharp.lowlight_loader(config.lowlight_images_path,config.overlight_images_path)    #dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    prompt_train_dataset = dataloader_prompt_margin.lowlight_loader(config.lowlight_images_path,config.normallight_images_path)#,config.overlight_images_path)        
    prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    prompt_train_dataset_1 = dataloader_prompt_add.lowlight_loader(config.lowlight_images_path,config.normallight_images_path)
    prompt_train_loader_1 = torch.utils.data.DataLoader(prompt_train_dataset_1, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    
    #loss
    text_encoder = TextEncoder(model)
    L_clip = clip_score.L_clip_from_feature()
    L_clip_MSE = clip_score.L_clip_MSE()
    L_margin_loss = clip_score.four_margin_loss(0.9,0.2)#0.9,0.2
    
    #load gradient update strategy.
    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)
    # reconsturction_train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.reconstruction_train_lr, weight_decay=config.weight_decay)
    prompt_optimizer = torch.optim.Adam(learn_prompt.parameters(), lr=config.prompt_lr, weight_decay=config.weight_decay)

    #initial parameters
    U_net.train()
    total_iteration=0
    cur_iteration=0
    max_score_psnr=-10000
    pr_last_few_iter=0
    score_psnr=[0]*30
    semi_path=['','']
    pr_semi_path=0
    #last_iteration=0
    best_model=U_net
    best_prompt=learn_prompt
    min_prompt_loss=100
    best_prompt_iter=0
    best_model_iter=0
    rounds=0
    reconstruction_iter=0
    reinit_flag=0
    
    #Start training!
    for epoch in range(config.num_epochs):
        if total_iteration<config.num_clip_pretrained_iters:
            train_thre=0
            total_thre=config.num_clip_pretrained_iters
        elif total_iteration<config.num_reconstruction_iters+config.num_clip_pretrained_iters:
            train_thre=config.num_reconstruction_iters
            total_thre=config.num_reconstruction_iters
        elif cur_iteration==0:
            train_thre=2100#800#2100#800#200
            total_thre=3100#2800#3100#1200#500
            print("cur using prompt from: iteration ", best_prompt_iter)
            print("cur using best model from: iteration ", best_model_iter)
        if cur_iteration+1<=train_thre: 
            if cur_iteration==0:
                learn_prompt=best_prompt
            embedding_prompt=learn_prompt.module.embedding_prompt
            embedding_prompt.requires_grad = False
            tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*config.length_prompt)]])
            text_features = text_encoder(embedding_prompt,tokenized_prompts)
            #fix the prompt and train the enhancement model
            for name, param in learn_prompt.named_parameters():
                param.requires_grad_(False)

            for iteration, item in enumerate(train_loader): 
        
                img_lowlight ,img_lowlight_path=item
                
                img_lowlight = img_lowlight.cuda()

                light_map  = U_net(img_lowlight)
                final=torch.clamp(((img_lowlight) /(light_map+0.000000001)),0,1)
               
                cliploss=16*20*L_clip(final, text_features)
                clip_MSEloss = 25*L_clip_MSE(final, img_lowlight,[1.0,1.0,1.0,1.0,0.5])

                if(total_iteration>=config.num_reconstruction_iters+config.num_clip_pretrained_iters):
                    # print("training model with cliploss and reconstruction loss")
                    loss = cliploss + 0.9*clip_MSEloss
                else:
                    # print("reconstruction...")
                    loss = 25*L_clip_MSE(final, img_lowlight,[1.0,1.0,1.0,1.0,1.0])
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()
                with torch.no_grad():
                    if total_iteration<config.num_reconstruction_iters+config.num_clip_pretrained_iters:
                        score_psnr[pr_last_few_iter] = torch.mean(iqa_metric(img_lowlight, final))
                        reconstruction_iter+=1
                        if sum(score_psnr).item()/30.0 < 8 and reconstruction_iter >100:
                            reinit_flag=1
                    else:
                        score_psnr[pr_last_few_iter] = -loss

                    pr_last_few_iter+=1
                    if pr_last_few_iter==30:
                        pr_last_few_iter=0
                    if (sum(score_psnr).item()/30.0)>max_score_psnr and ((total_iteration+1) % config.display_iter) == 0:
                        max_score_psnr=sum(score_psnr).item()/30.0
                        torch.save(U_net.state_dict(), config.train_snapshots_folder + "best_model_round"+str(rounds) + '.pth')    
                        best_model=U_net
                        best_model_iter=total_iteration+1
                        print(max_score_psnr)
                        inference(config.lowlight_images_path,'./'+task_name+'/result_'+task_name+'/result_jt_'+str(total_iteration+1)+"_psnr_or_-loss"+str(max_score_psnr)[:8]+'/',U_net,256)
                        if total_iteration >config.num_reconstruction_iters+config.num_clip_pretrained_iters:
                            semi_path[pr_semi_path]='./'+task_name+'/result_'+task_name+'/result_jt_'+str(total_iteration+1)+"_psnr_or_-loss"+str(max_score_psnr)[:8]+'/'
                            print(semi_path)
                        torch.save(U_net.state_dict(), config.train_snapshots_folder + "iter_" + str(total_iteration+1) + '.pth')   
                if reinit_flag == 1:
                    print(sum(score_psnr).item()/30.0)
                    print("reinitialization...")
                    seed=random.randint(0,100000)
                    print("current random seed: ",seed)
                    torch.cuda.manual_seed_all(seed)
                    U_net=model_small.UNet_emb_oneBranch_symmetry_noreflect(3,1).cuda()
                    U_net.apply(weights_init)
                    U_net= torch.nn.DataParallel(U_net)
                    reconstruction_iter=0
                    train_optimizer = torch.optim.Adam(U_net.parameters(), lr=config.train_lr, weight_decay=config.weight_decay)
                    config.num_reconstruction_iters+=100
                    reinit_flag=0
                
                if ((total_iteration+1) % config.display_iter) == 0:
                    print("training current learning rate: ",train_optimizer.state_dict()['param_groups'][0]['lr'])
                    print("Loss at iteration", total_iteration+1,"epoch",epoch, ":", loss.item())
                    print("loss_clip",cliploss," reconstruction loss",clip_MSEloss)
                    writer.add_scalars('Loss_train', {'train': loss,"clip": cliploss," reconstruction loss":clip_MSEloss},total_iteration+1)
                    print(cur_iteration+1," ",total_iteration+1)
                    print(train_thre,' ',total_thre)

                if cur_iteration+1==train_thre and total_iteration>config.num_reconstruction_iters+config.num_clip_pretrained_iters and (cliploss+0.9*clip_MSEloss>config.thre_train):
                    train_thre+=60
                    total_thre+=60
                elif cur_iteration+1==train_thre:
                    cur_iteration+=1
                    total_iteration+=1
                    print("switch to fine-tune the prompt pair")
                    break
                cur_iteration+=1
                total_iteration+=1
            embedding_prompt.requires_grad =True
        elif cur_iteration+1>=total_thre:
            cur_iteration=0
            train_thre=0
            total_thre=0
            min_prompt_loss=100
            max_score_psnr=-10000
            score_psnr=[0]*30
            rounds+=1
        else:
            #prompt initialization
            if total_iteration<config.num_clip_pretrained_iters:
                for name, param in U_net.named_parameters():
                    param.requires_grad_(False)
                for iteration, item in enumerate(prompt_train_loader_1):
                    img_lowlight,label=item    
                    img_lowlight = img_lowlight.cuda()
                    label = label.cuda()
                    output=learn_prompt(img_lowlight,0)
                    loss=F.cross_entropy(output,label)
                    prompt_optimizer.zero_grad()
                    loss.backward()
                    prompt_optimizer.step()
                    
                    if ((total_iteration+1) % config.prompt_display_iter) == 0:
                        if loss<min_prompt_loss:
                            min_prompt_loss=loss
                            best_prompt=learn_prompt
                            best_prompt_iter=total_iteration+1
                            torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "best_prompt_round"+str(rounds) + '.pth')
                        print("prompt current learning rate: ",prompt_optimizer.state_dict()['param_groups'][0]['lr'])
                        print("Loss at iteration", total_iteration+1, ":", loss.item())
                        print("output",output.softmax(dim=-1),"label",label)
                        print("cross_entropy_loss",loss)
                        writer.add_scalars('Loss_prompt', {'train':loss}, total_iteration)
                        print(cur_iteration+1," ",total_iteration+1)
                        print(train_thre,' ',total_thre)
                    if ((total_iteration+1) % config.prompt_snapshot_iter) == 0:
                        torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "iter_" + str(total_iteration+1) + '.pth')  
                    if cur_iteration+1==total_thre and loss>config.thre_prompt:#loss>last_prompt_loss[flag_prompt]*0.95:#loss>0.01:#
                        #train_thre+=20
                        total_thre+=100
                    elif cur_iteration+1==total_thre:
                        cur_iteration+=1
                        total_iteration+=1
                        break
                    cur_iteration+=1
                    total_iteration+=1     
            else:
                #prompt fine-tuning
                if cur_iteration+1==train_thre+1:
                    if total_iteration+1>config.num_clip_pretrained_iters:
                        pr_semi_path=1-pr_semi_path
                    U_net=best_model
                    if semi_path[0]=='':
                        print(semi_path)
                        L_margin_loss = clip_score.four_margin_loss(1.0,0.2)
                    elif semi_path[1]=='':
                        print(semi_path)
                        L_margin_loss = clip_score.four_margin_loss(0.9,0.2)
                        prompt_train_dataset = dataloader_prompt_margin.lowlight_loader(config.lowlight_images_path,config.normallight_images_path,semi_path[0])#,config.overlight_images_path)        
                        prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
                    else:
                        print(semi_path)
                        L_margin_loss = clip_score.four_margin_loss(0.9,0.1)
                        prompt_train_dataset = dataloader_prompt_margin.lowlight_loader(config.lowlight_images_path,config.normallight_images_path,semi_path[1-pr_semi_path],semi_path[pr_semi_path])#,config.overlight_images_path)        
                        prompt_train_loader = torch.utils.data.DataLoader(prompt_train_dataset, batch_size=config.prompt_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
                
                #fix enhancement model and train the prompt
                for name, param in U_net.named_parameters():
                    param.requires_grad_(False)
                    
                for iteration, item in enumerate(prompt_train_loader):
                    img_feature_list,labels=item 
                    labels=labels.cuda()
                    if len(img_feature_list)==2:
                        inp,ref=img_feature_list
                        loss=200*L_margin_loss(learn_prompt(inp.cuda()),learn_prompt(ref.cuda()),labels,2)
                    elif len(img_feature_list)==3:
                        inp,semi1,ref=img_feature_list
                        loss=200*L_margin_loss(learn_prompt(inp.cuda()),learn_prompt(ref.cuda()),labels,3,learn_prompt(semi1.cuda()))
                    else:
                        inp,semi1,semi2,ref = img_feature_list
                        loss=200*L_margin_loss(learn_prompt(inp.cuda()),learn_prompt(ref.cuda()),labels,4,learn_prompt(semi1.cuda()),learn_prompt(semi2.cuda()))
                    prompt_optimizer.zero_grad()
                    loss.backward()
                    prompt_optimizer.step()
                    if ((total_iteration+1) % config.prompt_display_iter) == 0:
                        if loss<min_prompt_loss:
                            min_prompt_loss=loss
                            best_prompt=learn_prompt
                            best_prompt_iter=total_iteration+1
                            torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "best_prompt_round"+str(rounds) + '.pth')
                        print("prompt current learning rate: ",prompt_optimizer.state_dict()['param_groups'][0]['lr'])
                        print("Loss at iteration", total_iteration+1, ":", loss.item())
                        print("margin_loss",loss)
                        writer.add_scalars('Loss_prompt', {'train':loss}, total_iteration)
                        print(cur_iteration+1," ",total_iteration+1)
                        print(train_thre,' ',total_thre)
                
                    if ((total_iteration+1) % config.prompt_snapshot_iter) == 0:
                        torch.save(learn_prompt.state_dict(), config.prompt_snapshots_folder + "iter_" + str(total_iteration+1) + '.pth')     
                    if cur_iteration+1==total_thre and loss>config.thre_prompt :#and ((total_thre-train_thre)<3000):#loss>last_prompt_loss[flag_prompt]*0.95:#loss>0.01:#
                        #train_thre+=20
                        total_thre+=100
                    elif cur_iteration+1==total_thre:
                        cur_iteration+=1
                        total_iteration+=1
                        print("switch to tuning the enhancement model")
                        break
                    cur_iteration+=1
                    total_iteration+=1
            for name, param in U_net.named_parameters():
                param.requires_grad_(True)
            

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('-b','--lowlight_images_path', type=str, default="./train_data/BAID_380/resize_input/") 
    parser.add_argument('--overlight_images_path', type=str, default=None)
    parser.add_argument('-r','--normallight_images_path', type=str, default='./train_data/DIV2K_384/') 
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--thre_train', type=float, default=90)
    parser.add_argument('--thre_prompt', type=float, default=60)
    parser.add_argument('--reconstruction_train_lr',type=float,default=0.00005)#0.0001
    parser.add_argument('--train_lr', type=float, default=0.00002)#0.00002#0.00005#0.0001
    parser.add_argument('--prompt_lr', type=float, default=0.000005)#0.00001#0.00008
    parser.add_argument('--T_max', type=float, default=100)
    parser.add_argument('--eta_min', type=float, default=5e-6)#1e-6
    parser.add_argument('--weight_decay', type=float, default=0.001)#0.0001
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=2000)#3000
    parser.add_argument('--num_reconstruction_iters', type=int, default=0)#1000
    parser.add_argument('--num_clip_pretrained_iters', type=int, default=0)#8000
    parser.add_argument('--noTV_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--prompt_batch_size', type=int, default=16)#32
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=20)
    parser.add_argument('--snapshot_iter', type=int, default=20)
    parser.add_argument('--prompt_display_iter', type=int, default=20)
    parser.add_argument('--prompt_snapshot_iter', type=int, default=100)
    parser.add_argument('--train_snapshots_folder', type=str, default="./"+task_name+"/"+"snapshots_train_"+task_name+"/")
    parser.add_argument('--prompt_snapshots_folder', type=str, default="./"+task_name+"/"+"snapshots_prompt_"+task_name+"/")
    parser.add_argument('--load_pretrain', type=lambda x: (str(x).lower() == 'true'), default= True)
    parser.add_argument('--pretrain_dir', type=str, default= './pretrained_models/init_pretrained_models/init_enhancement_model.pth')
    parser.add_argument('--load_pretrain_prompt', type=lambda x: (str(x).lower() == 'true'), default= True)
    parser.add_argument('--prompt_pretrain_dir', type=str, default= './pretrained_models/init_pretrained_models/init_prompt_pair.pth')
    
    config = parser.parse_args()

    if not os.path.exists(config.train_snapshots_folder):
        os.mkdir(config.train_snapshots_folder)
    if not os.path.exists(config.prompt_snapshots_folder):
        os.mkdir(config.prompt_snapshots_folder)
  

    train(config)
