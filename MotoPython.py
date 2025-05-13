Print('Starting python file for the project')

#!/usr/bin/env python
# coding: utf-8

# In[41]:


# create graphs  to calculate CD
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Vx=[20,30,40,50,60,70] #velocity  data points
Fd1=[3,6,10.5,16.5,22,31] #FD of mountian bike
Fd2=[2,4,8,13,17,23]  #FD of road race bike
Fd3=[1.5,3,6,10.5,14,19.5] #FD of time trial bike

plt.scatter(Vx,Fd1,label="Mountain bike", color='k', marker='D', edgecolors='yellow', alpha=.75)

plt.scatter(Vx,Fd2,label="Road Race Bike", color='k', marker="^",edgecolors='red', alpha=.75)
plt.scatter(Vx,Fd3,label="Time trial Bike", color='k', marker="o", edgecolors='blue', alpha=.75)
plt.xlabel('Speed (Km/Hr)', fontsize=12, color='blue')
plt.ylabel('Force Drag (N)', fontsize=12, color='blue')

def bikedrag(V,drag,A=.018, rho=1.225): #model to generate fitted lines of data points
    rez=[]
    for i in range(len(V)):
        CofD=((2*drag[i])/(1.225*.018*(V[i]**2)))
        rez.append(CofD)
    C_D=mean(Rez)
    return C_D



def drforcebike(S,CD): #formula to detemine coeffecients of drag
    res=[]
    for i in range(len(S)):
        Forcedrag=0.5 * .018 * CD * 1.225 * S[i]**2
        res.append( Forcedrag)
    return res



# Curve fitting (fix A and rho, fit only Cd)
def modelfit(V, Cd):
    return drforcebike(Vx,Cd)


popt1,pcov1 = curve_fit(modelfit, Vx, Fd1, p0=[1.1]) #using curve fit to find drag coeffecients
Cd_mb = popt1[0]
mberr=float(np.sqrt(np.diag(pcov1)))

popt2,pcov2 = curve_fit(modelfit, Vx, Fd2, p0=[1.1])
Cd_rr = popt2[0]
rrerr=float(np.sqrt(np.diag(pcov2)))

popt3,pcov3 = curve_fit(modelfit, Vx, Fd3, p0=[1.1])
Cd_tt = popt3[0]
tterr=float(np.sqrt(np.diag(pcov3)))
print(f"Estimated Drag Coefficient for Mountain Bike(Cd): {Cd_mb:.4f}  ± {mberr:.4f} ")
print(f"Estimated Drag Coefficient (Cd)for Road race Bike: {Cd_rr:.4f}  ± {rrerr:.4f} ")
print(f"Estimated Drag Coefficient (Cd)for Time Trial Bike: {Cd_tt:.4f}  ± {tterr:.4f} ")

# creating line of fits for each data set
xfit=np.linspace(0,80,800)
yfit1=drforcebike(xfit,Cd_mb)
yfit2=drforcebike(xfit,Cd_rr)
yfit3=drforcebike(xfit,Cd_tt)

plt.plot(xfit,yfit1, '-y', label='Fitted Curve of Mountain bike ')
plt.plot(xfit,yfit2, '-r', label='Fitted curve of Road Race Bike')
plt.plot(xfit,yfit3, '-b', label='Fitted curve of Time Trial bike')
plt.legend()
plt.grid(True)
plt.xlim(0, 80);
plt.ylim(0, 35);
plt.yticks([0,5,10,15,20,25,30,35]);
plt.xticks([0,20,40,60,80]);




# In[35]:


import numpy as np
import matplotlib.pyplot as plt

def Topspeed(P_engine_watts, Cd, A, rho=1.225):
    """
    Calculate top speed from power, Cd, frontal area, and air density.
    """
    return ((2 * P_engine_watts) / (rho * Cd * A)) ** (1/3)

# Parameters
Cd = 0.5     # Drag coefficient (random number)
A = 0.6      # Frontal area (m²)
rho = 1.225  # Air density (kg/m³)

# Engine powers (W): from 10 hp to 200 hp (1 hp ≈ 745.7 W)
powers_hp = np.linspace(10, 200, 100)
powers_watts = powers_hp * 745.7

# Calculate corresponding top speeds
speeds_mps = Topspeed(powers_watts, Cd, A, rho)
speeds_kph = speeds_mps * 3.6
drc=str(Cd)
# Plot
plt.figure(figsize=(8, 5))
plt.plot(powers_hp, speeds_kph, color='darkblue', linewidth=2)
plt.xlabel("Engine Power (hp)")
plt.ylabel("Top Speed (km/h)")
plt.title("Top Speed vs Engine Power for Motorcycle for Drag coeffecient: "+ drc)
plt.grid(True)
plt.tight_layout()
plt.show()

