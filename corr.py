import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

with open("parmesh.csv", mode='r', newline='') as f_in:
    reader = csv.reader(f_in)
    data_array = [row for row in reader]
m=np.array(data_array)
m=m.astype(np.float64)
y=np.zeros((31,25))
#for i in range(25):
  #  y[:,i]=m[:,i]
#sig1=m[:,1-1]
#sig2=m[:,5-1]
plt.plot(np.arange(31), m[:,12], marker="o", color = "red", linestyle = "-",label="x = area12")
plt.plot(np.arange(31)-1, m[:,13], marker="o", color = "blue", linestyle = "--",label="y = area13")
#plt.legend()
plt.show()
#print(m[:,11])
#print(m[:,12])

for i in range(25):
	for j in range(25):
		if i<j:
			plt.xlabel('t(lag)')
			plt.title('Cross-correlation between %d and %d'%(i+1,j+1))
			sig1=m[:,i+1]
			sig2=m[:,j+1]
			#sig1=sig1-sig1.mean()
			#sig2=sig2-sig2.mean()
			#sig1=sig1/sig1.max()
			#sig2=sig2/sig2.max()
			
			corr=np.correlate(sig1,sig2,"full")
			#print(m)
			print(corr)
			plt.grid(True)	
			plt.ylabel("Correlation coefficient")
			#plt.plot(np.arange(len(corr)) - 31 + 1, corr, color="r")
			plt.xcorr(sig1, sig2, usevlines=True, maxlags=30, normed=True, lw=2,color="black",linestyle="-")
			plt.xlim([-31, 31])
			#plt.savefig('images/corr_normed/%d-%d.png'%(i+1,j+1))
			plt.show()
