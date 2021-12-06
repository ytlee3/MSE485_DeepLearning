import numpy as np
import random
mask64 = (1 << 64) - 1
mask32 = (1 << 32) - 1
def my_LCG(m,a,c,x0,N):
    rand = np.zeros(N)
    rand[0] = (a*x0+c)%m
    for i in np.arange(1,N):
        rand[i] = (a*rand[i-1]+c)%m
    return rand
def my_MRG(m,a1,ak,N):
    rand = np.zeros(N)
    rand[0] = (a1*2+ak*4)%m
    rand[1] = (a1 * rand[0] + ak * 3) % m
    rand[2] = (a1 * rand[1] + ak * 5) % m
    for i in np.arange(3,N):
        rand[i] = (a1*rand[i-1]+ak*rand[i-3])%m
    return np.divide(rand,m)
def my_LFG(m,N,j,k):
    rand = np.zeros(N)
    for i in np.arange(k):
        rand[i] = int(random.random()*m)
        print("LFG init cond: " +str(rand[i]))
    for i in np.arange(k,N):
        rand[i] = (rand[i-j]+rand[i-k])%m
    return np.divide(rand, m)
def rotate_right(x, n):
    return int(f"{x:032b}"[-n:] + f"{x:032b}"[:-n], 2)
def my_PCG(m,a,c,x0,N):
    rand = np.zeros(N)
    rand[0] = int((a*x0+c)%m)
    for i in np.arange(1,N):
        state = int(a*rand[i-1]+c) & mask64
        rand[i] =  rotate_right(state, (29 - (rotate_right(state,61)))) & mask32
    return rand

def my_CheckRandomNumbersND(rand_array,NB,ND):
    """
    This function can be reused later for larger data set

    Input:
    rand_array: array of size N-by-3, so (rand_array[0][0], rand_array[0][1], rand_array[0][2]) is the first 3D point
    NB: number of bins per dimension (for 3D we need NB*NB*NB bins in total)

    Output:
    the chi-squared value of the rand_array, using NB*NB*NB evenly spaced bins in [0,1)x[0,1)x[0,1).
    """
    print((((0,1),)*ND))
    exp_counts = np.shape(rand_array)[0]/(NB**ND)
    hist = np.histogramdd(rand_array,bins = NB,range = ((0,1),)*ND)[0]
    return np.sum(np.divide(np.square(np.subtract(np.ravel(hist),exp_counts)),exp_counts))

#pyrand = []
#for i in range(10000):
#    pyrand.append(random.random())
#pyrand = np.array(pyrand)
#pyrand = pyrand.reshape(5000,2)
#myrand = my_MRG(2**31 - 21069,2197254,-1967928,2**18)
#myrand = my_PCG(2**32,319993,1,0,2**18) / 2**32
myrand = my_LFG(2**32,2**18,7,10)
#myrand = my_LCG(2**48,25214903917,11,0,2**18) / 2**48
#myrand = my_LCG(2**31,134775813,1,0,2**18)/2**31
#myrand = np.loadtxt('Qrand_2_18.txt')
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

"""LFG init cond: 3497106342.0
LFG init cond: 1726050929.0
LFG init cond: 2764847479.0
LFG init cond: 3019656476.0
LFG init cond: 1047210913.0
LFG init cond: 3142016416.0
LFG init cond: 2761121183.0
LFG init cond: 3614129048.0
LFG init cond: 4138112094.0
LFG init cond: 2410941971.0"""