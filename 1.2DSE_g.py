import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
##Define N and dy, Ndy=1 > dy=1/N
N = 1000
dy = 1/N
y = np.linspace(0,1, N+1) #N+1 for psi0 and psiN
##define mL^2V (mL2>1000 for unbounded problem)
mL2=500
a=5
u=2.5
sigm=0.5
def mL2V(y):
    return mL2*((1/np.sqrt(2*np.pi*sigm**2))*np.e**(((-1/2)*(a*y-u)**2)/sigm**2))

V =mL2V(y)

##shape of V
plot0 = plt.figure(0)
plt.plot(y, V)
plt.title("Plot of the shape of potential term ")

legend1=plt.legend([r'$V_G(y)= \frac{1} {\sigma\sqrt{2 \pi}}e^{\frac{-(ay-u)^2}{2\sigma^2}}$'], loc =1)
ax = plt.gca().add_artist(legend1)

legend2=plt.legend(['$a={}, u={}, \sigma={}$'.format(a, u, sigm)], loc =4, handlelength=0)
ax = plt.gca().add_artist(legend2)


##Creta matrix
md = 1/(dy**2) + mL2V(y)[1:-1] #no 1st and last point
upd = -1/(2*dy**2)* np.ones(len(md)-1) #upper d

Energy , state = eigh_tridiagonal(md, upd) #Solve eigenvalue problem for a real symmetric tridiagonal matrix.

#plt.figure(dpi=120)

##plot eignstate
plot1 = plt.figure(1)

plt.plot(state.T[0])
plt.plot(state.T[1])
plt.plot(state.T[2])
plt.plot(state.T[3])

plt.title("Plot of eigenfunctions for different $\psi$ ")
plt.xlabel(r'Position $x$')
plt.ylabel(r'Eigenfunctions $\psi$')

legend1=plt.legend(['$\psi_1$','$\psi_2$','$\psi_3$','$\psi_4$'], loc =1)
ax = plt.gca().add_artist(legend1)

legend2=plt.legend(['$N={}$'.format(N),'$dy=1/N={}$'.format(dy),'$mL^2={}$'.format(mL2)], loc =4, handlelength=0)
ax = plt.gca().add_artist(legend2)

##plot probability
plot2 = plt.figure(2)

plt.plot(state.T[0]**2)
plt.plot(state.T[1]**2)
plt.plot(state.T[2]**2)
plt.plot(state.T[3]**2)

plt.title("Plot of probability density for different $\psi$ ")
plt.xlabel(r'Position $x$')
plt.ylabel(r'Probability $\rho$')

legend1=plt.legend(['$\psi_1^2$','$\psi_2^2$','$\psi_3^2$','$\psi_4^2$'], loc =1)
ax = plt.gca().add_artist(legend1)

legend2=plt.legend(['$N={}$'.format(N),'$dy=1/N={}$'.format(dy),'$mL^2={}$'.format(mL2)], loc =4, handlelength=0)
ax = plt.gca().add_artist(legend2)

##plot Energy
plot3 = plt.figure(3)
plt.scatter(np.arange(0,10,1), Energy[0:10], s=1444, marker="_", linewidth=2, zorder=3)

plt.title("Plot of eignvalues")
plt.xlabel(r'N')
plt.ylabel(r'$mL^2 E/\hbar^2$')

n = ['$E_{}$'.format(i) for i in range(0,10)]

for i, txt in enumerate(n):
    plt.annotate(txt, (np.arange(0,10,1)[i], Energy[0:10][i]), ha="center")

plt.show()


