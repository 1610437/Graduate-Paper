import csv
import numpy as np
import numpy.matlib
import tqdm

x1=np.array([ v for v in csv.reader(open("./smoothed_inf_large.csv", "r")) if len(v)!= 0])
#x1=np.array([ v for v in csv.reader(open("flow2L.csv", "r")) if len(v)!= 0])
#code=np.array([ v for v in csv.reader(open("../大阪市町村コード.csv", "r")) if len(v)!= 0])
x1=x1.astype(np.float64)
print(x1.shape)
permesh=np.zeros((151,25))

for i in tqdm.trange(x1.shape[1]):
	if i in np.array([31]):
		 permesh[:,0]=permesh[:,0]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([30,33,35]):
		 permesh[:,1]=permesh[:,1]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([46]):
		 permesh[:,2]=permesh[:,2]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([43]):
		 permesh[:,3]=permesh[:,3]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([66]):
		 permesh[:,4]=permesh[:,4]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([63]):
		 permesh[:,5]=permesh[:,5]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([15]):
		 permesh[:,6]=permesh[:,6]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([6,7,8,9,10,11,26]):
		 permesh[:,7]=permesh[:,7]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([28,29,45]):
		 permesh[:,8]=permesh[:,8]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([83,84]):
		 permesh[:,9]=permesh[:,9]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([57,64,65]):
		 permesh[:,10]=permesh[:,10]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([47,48,49,50,51,52,53,58,59,60,61,62,67,68]):
		 permesh[:,11]=permesh[:,11]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([69,70,71,72,73,74]):
		 permesh[:,12]=permesh[:,12]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([1,12,13,14,16,17,75]):
		 permesh[:,13]=permesh[:,13]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([0,2,3,4,5,20,24]):
		 permesh[:,14]=permesh[:,14]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([27]):
		 permesh[:,15]=permesh[:,15]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([42]):
		 permesh[:,16]=permesh[:,16]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([54,55,56]):
		 permesh[:,17]=permesh[:,17]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([78]):
		 permesh[:,18]=permesh[:,18]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([76,77,79]):
		 permesh[:,19]=permesh[:,19]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([18,19,21]):
		 permesh[:,20]=permesh[:,20]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([22,23]):
		 permesh[:,21]=permesh[:,21]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([35]):
		 permesh[:,22]=permesh[:,22]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([36]):
		 permesh[:,23]=permesh[:,23]+x1[:,i]
		 
for i in tqdm.trange(x1.shape[1]):
	if i in np.array([37,38,40]):
		 permesh[:,24]=permesh[:,24]+x1[:,i]
		 
for i in range(25):
	permesh[:,i]=permesh[:,i]/permesh[:,i].max()
	print(permesh[:,i].min()) 
f = open('parmesh_norm_large.csv', 'w')
writer = csv.writer(f, lineterminator='\n')
for i in tqdm.trange(permesh.shape[0]):
	writer.writerow(permesh[i,:])
f.close()
		 
