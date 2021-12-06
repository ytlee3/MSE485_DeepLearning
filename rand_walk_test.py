import numpy as np
import random
d = 2**18
k=2**8
count1 = 0
count2 = 0
count3 = 0
count4 = 0
countonorigin = 0
#np.array([[.4,0,.6],[.5,.3.2],[0,1,.3]])
for i in np.arange(int(d/k)):
    x=0
    y=0
    for j in np.arange(k):
        p=myrand[k*i+j]
        if(p<.25):
            x+=1
        elif(p<.5):
            x-=1
        elif(p<.75):
            y+=1
        else:
            y-=1
    if(y >=0 and x> 0):
        count1 +=1
    elif(y>0 and x<=0):
        count2 +=1
    elif(y<=0 and x<0):
        count3 +=1
    elif(y<0 and x>=0):
        count4+=1
    else:
        countonorigin +=1


count1+= countonorigin/4
count2+= countonorigin/4
count3+= countonorigin/4
count4+= countonorigin/4

total = d
E = (d)/(4*k)
chisq = ((count1 - E)**2  + (count2 - E)**2 + (count3 - E)**2 + (count4 - E)**2)/E
print(chisq)
print(np.mean(myrand))
