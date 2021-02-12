from pathlib import Path
import tqdm
from collections import OrderedDict
import numpy as np

import torch
import torch.optim as optim

import tensorboardX

import model
import datas
import dataloader
import csv

stay_ratio=0.1
day=1

if __name__ == "__main__":
    #Select data which you use
    flowtype = 1
    #Hyper parameter for objective function
    lam=8.0
    #Dimention of input layer and hidden layer
    input_layer=5
    hidden_layer=40

    if flowtype==21:
        population_data, location_table, adj_table ,trans_prob= datas.read_kansai2Global()
        location = [[row[0] / 3 - 0.5, row[1] / 2 - 0.5] for row in location_table]
    elif flowtype==22:
        population_data, location_table, adj_table ,trans_prob= datas.read_kansai2Local()
        location = [[row[0] / 9 - 0.5, row[1] / 9 - 0.5] for row in location_table]
    elif flowtype==1:
        population_data, location_table, adj_table ,trans_prob= datas.read_kansai1G()
        location = [[row[0] / 9 - 0.5, row[1] / 11 - 0.5] for row in location_table]

    trans_prob=trans_prob*0.01
    time_size = population_data.shape[0]
    location_size = population_data.shape[1]
    z_tensor=torch.zeros(time_size-1,location_size,location_size,dtype=torch.double)
    for i in range(5):
        print(population_data[3,:]/10**(i+1))

    c=0
    for l in tqdm.trange(time_size-1):
        for ll in range(location_size):
            if 0<=c and c<6:
                z_tensor[l,ll,:]=adj_table[ll,]*population_data[l,ll]*trans_prob[c%6,1]#/adj_table[ll,].sum()
                z_tensor[l,ll,ll]=population_data[l,ll]*adj_table[ll,ll]*(1-trans_prob[l%6,1])
            elif 6<=c and c<12:
                z_tensor[l,ll,:]=adj_table[ll,]*population_data[l,ll]*trans_prob[l%6,2]#/adj_table[ll,].sum()
                z_tensor[l,ll,ll]=population_data[l,ll]*adj_table[ll,ll]*(1-trans_prob[l%6,2])
            elif 12<=c and c<42:
                z_tensor[l,ll,:]=adj_table[ll,]*population_data[l,ll]*trans_prob[l%6,0]#/adj_table[ll,].sum()
                z_tensor[l,ll,ll]=population_data[l,ll]*adj_table[ll,ll]*(1-trans_prob[l%6,0])
        print(trans_prob[c%6,day])
        c=c+1
        if c==41:
            c=0

    f = open('outputcsv/zinitLocal.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    for l in range(time_size-1):
        for ll in range(location_size):
           writer.writerow(z_tensor[l,ll,:].detach().numpy())
    f.close()
    #Use cuda
    use_cuda = True
    available_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if (use_cuda and available_cuda) else 'cpu')
    print(device)
    #Set default type of tensor
    torch.set_default_dtype(torch.double)
    #torch.set_grad_enabled(True)
    #torch.autograd.set_detect_anomaly(True)

    #Use tensorboardX
    board = tensorboardX.SummaryWriter()

    #Instantinate model
    mod = model.NCGM(input_layer, hidden_layer,z_tensor)
    mod.to(device)

    #Instantinate objective function
    objective = model.NCGM_objective(location_size,adj_table)

    #Instantinate optimizer
    #optimizer = optim.SGD(mod.parameters(), lr=0.5)
    optimizer = optim.Adam(mod.parameters())

    #Instantinate dataloader
    data_loader = dataloader.Data_loader(population_data, location, time_size, location_size, device)

    #Training
    mod.train()
    itr = tqdm.trange(2000)
    #itr = tqdm.trange(1)
    losses = []
    ave_loss = 0.0
    for i in itr:
        for t in range(time_size - 1):
        #for t in range(1):
            input_data, yt, yt1 = data_loader.get_t_input(t)
            theta = mod(input_data)
            loss = objective(theta, mod.Z[t], yt, yt1, lam)
            #print(loss)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item(), b_grad=mod.fc2.bias.grad))
            #itr.set_postfix(ordered_dict=OrderedDict(loss=loss.item()))

            board.add_scalar("loss", loss.item(), i * (time_size - 1) + t)
            ave_loss = ave_loss + loss.item()


        itr.set_postfix(ordered_dict=OrderedDict(loss=ave_loss/(time_size-1),))
        board.add_text("Z", str(mod.Z), i)
        board.add_scalar("ave_loss", ave_loss / (time_size - 1), i)
        ave_loss = 0.0

        #with open("output/{0:05}.txt".format(i), 'wt') as f:
          #  f.write(str(mod.Z.data.numpy()))

    #tensorboard用の値のjsonファイルへの保存[ポイント6]
    board.export_scalars_to_json("./all_scalars.json")
    board.add_text("progress", "finish", 0)
    #SummaryWriterのclose[ポイント7]
    board.close()
    #mod.Z=mod.Z.clamp(min=1.0e-10)

    if flowtype==1:
        f = open('outputcsv/flow1/zG.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        for l in range(time_size-1):
            for ll in range(location_size):
                if ll in np.array([7,8,9,10,12,13,14,15,16,20,27,28,29,30,31,32,33,36,37,38,39,40,48,51,54]):
                    writer.writerow(mod.Z[l,ll,:].detach().numpy()*model.digit)
        #writer.writerow(np.array(range(1,57)))
        f.close()

        f = open('outputcsv/flow1/thetaG.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        for l in range(location_size):
            if ll in np.array([7,8,9,10,12,13,14,15,16,20,27,28,29,30,31,32,33,35,36,37,38,39,40,48,51,54]):
                writer.writerow(theta[l,:].detach().numpy())
        f.close()
    if flowtype==21:
        f = open('outputcsv/flow2/zGlobal.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        for l in range(time_size-1):
            for ll in range(location_size):
                writer.writerow(mod.Z[l,ll,:].detach().numpy()*model.digit)
        f.close()

        f = open('outputcsv/flow2/thetaGlobal.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        for l in range(location_size):
            writer.writerow(theta[l,:].detach().numpy())
        f.close()

    elif flowtype==22:
        f = open('outputcsv/flow2/zLocal2km.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        for l in range(time_size-1):
            for ll in range(location_size):
                writer.writerow(mod.Z[l,ll,:].detach().numpy()*model.digit)
        f.close()

        f = open('outputcsv/flow2/thetaLocal2km.csv', 'w')
        writer = csv.writer(f, lineterminator='\n')
        for l in range(location_size):
            writer.writerow(theta[l,:].detach().numpy())
        f.close()

    x1=np.array([ v for v in csv.reader(open("outputcsv/flow1/zG.csv", "r")) if len(v)!= 0])
    #print(x1)
    x1=x1.astype(np.float64)

    for i in reversed(np.array([0,1,2,3,4,5,6,11,17,18,19,21,22,23,24,25,26,34,35,41,42,43,44,45,46,47,49,50,52,53,55,56,57,58])):
        x1=np.delete(x1,i,1)
    print(x1.shape)
    for i in range(x1.shape[0]):ax2 = ax1.twinx()
        for j in range(x1.shape[1]):
            if x1[i,j]<0:
                x1[i,j]=0
                continue;
    f = open('outputcsv/flow1/zG.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    for ll in range(x1.shape[0]):
        writer.writerow(x1[ll,:])
    f.close()
