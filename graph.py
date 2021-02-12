import csv
import matplotlib.pyplot as plt
import numpy as np
N=1
# figureの初期化
fig = plt.figure()
CELL=0
TIME=8

fn = 'parmesh.csv'
with open(fn, mode='r', newline='') as f_in:
    reader = csv.reader(f_in)
    data_array = [row for row in reader]
m=np.array(data_array)
m=m.astype(np.float64)
#print(data)

with open("./outputcsv/flow1/zG.csv", mode='r', newline='') as f_in:
    reader = csv.reader(f_in)
    data_array = [row for row in reader]
p=np.array(data_array)
p=p.astype(np.float64)

with open("./setdata/flow1/neighbourG.csv", mode='r', newline='') as f_in:
    reader = csv.reader(f_in)
    data_array = [row for row in reader]
n=np.array(data_array)
n=n.astype(np.float64)

x=m[:,0]
y=np.zeros((31,25))
flow=np.zeros((31,25))


n1=np.array([1,6,7])
n2=np.array([0,2,6,7,8])
n3=np.array([1,3,7,8])
n4=np.array([2,8])
n5=np.array([5,9,10,11])
n6=np.array([4,10,11,12])
n7=np.array([0,1,7,12,13,14])
n8=np.array([0,1,2,6,8,13,14,15])
n9=np.array([1,2,3,7,14,15,16])
n10=np.array([4,10])
n11=np.array([4,5,9,11,17])
n12=np.array([4,5,10,12,17,18])
n13=np.array([5,6,11,13,17,18,19])
n14=np.array([6,7,12,14,18,19,20])
n15=np.array([6,7,8,13,15,19,20,21])
n16=np.array([7,8,14,16,20,21])
n17=np.array([8,15,21])
n18=np.array([10,11,12,18])
n19=np.array([11,12,13,17,19])
n20=np.array([12,13,14,18,20])
n21=np.array([13,14,15,19,21,22])
n22=np.array([14,15,16,20,22])
n23=np.array([20,21,23])
n24=np.array([22,24])
n25=np.array([23])
N=n25
ii=25-1


for i in range(25):
    y[:,i]=m[:,i+1]
    print(y[:,i].max())
#------------------------------------------------------------------------------------------------------------------------------
#1からの人流
for i in range(30):
    for j in range(25):
        if j in N:
            flow[i,j]=p[i*25*6+25*(TIME-5)+ii,j]

for i in range(25):
    if flow[:,i].max()>0:
        print(flow[:,i].max())


#1への人流

for i in range(31):
    flow[i,0]=p[1+i*25*6+CELL+25*(TIME-5),0]
    flow[i,1]=p[6+i*25*6+CELL+25*(TIME-5),0]
    flow[i,2]=p[7+i*25*6+CELL+25*(TIME-5),0]




f = open('s.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
for ll in range(flow.shape[0]):
    writer.writerow(flow[ll,:])
f.close()




for j in N:
    plt.xlabel('Date')
    plt.ylabel('Smoothed prescriptions')
    plt.grid(True)
    plt.plot(x, y[:,ii], marker="o", color = "red", linestyle = "-",label="x = cell %d"%(ii+1))
    plt.plot(x, y[:,j], marker="o", color = "blue", linestyle = "--",label="y = cell %d"%(j+1))
    plt.legend()
    #plt.twinx().fill_between(x, flow[:,2], 0,facecolor='black',alpha=0.4)
    plt.twinx().bar(x, flow[:,j],facecolor='black',alpha=0.4)
    for k in np.array([1,2,8,9,15,16,22,23,24,29,30]):
        plt.bar(x[k-1], flow[k-1,j],facecolor='lime',alpha=0.4)
    #plt.legend()
    #plt.plot(x, y[:,N+3], marker="o", color = "yellow", linestyle = "--",label="y = area4")
    #plt.fill_between(x,y[:,1],0,facecolor='lime',alpha=1.0)
    plt.title('%d to %d'%(ii+1,j+1))

    #plt.twinx().set_ylabel('people flow')
    #plt.show()
    plt.savefig('images/toflow/%d-%d.png'%(ii+1,j+1))
    plt.show()




