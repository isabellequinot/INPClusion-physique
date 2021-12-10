# -*- coding: utf-8 -*-

"""
Regression avec une fonction en utilisant la methode du xi2 reduit
Prise en compte des incertitudes en X et en Y
"""

from scipy import *
from pylab import *
import scipy.optimize as spo

# Donnees a modifier par l'etudiant
# Abscisses et ordonnees
X = array([0.221, 0.325, 0.462, 0.562, 0.656, 0.810, 0.922, 0.980])
Y = array([2.172, 3.151, 4.542, 5.526, 6.434, 7.974, 9.001, 9.632])
# Incertitudes sur les abscisses et les ordonnees
uX = array([0.011, 0.016, 0.023, 0.028, 0.038, 0.041, 0.046,	0.047])
uY = array([0.107, 0.155, 0.421, 0.275, 0.321, 0.400, 0.450,	0.480])

##Regression
def modele(x,a): #modele lineaire
    return a*x

def derivee(x,a): #derivee du modele
    return a

# fonction d'ecart entre mesure et modele ponderee par les incertitudes expe.
def residual(a, x, y, ux, uy) :
    return (y-modele(x, a))/sqrt(uy**2+(derivee(x,a)*ux)**2)

#estimation des parametres à optimiser
p0=array([1]) 

results = spo.leastsq(residual, p0, args=(X,Y,uX,uY), full_output=True) 

a = results[0] #Parametres d'ajustement optimaux 
yth = modele(X, a) # donnees ajustees
chi2_red = sum(square( residual(a, X, Y, uX, uY)))/(X.size - 1) # evaluation du xi2 reduit                                      

# Matrice de variance covariance estimee
covm = results[1]   
# Erreur sur les parametres estimes
erra = sqrt(diag(covm))

#Graphique
fig, ax = plt.subplots(1)
ax.errorbar(X,Y,xerr=uX, yerr=uY,fmt='o',label="data")

ax.plot(X,yth,label="fit")
textstr = "$y = a \cdot x$ \n\
$a = %.3e \pm %.3e $\n\
$\chi_{2,red} = %.2e$"  %(a,erra,chi2_red)
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top')

title(r'Titre du graphique',fontsize=16)

xticks(fontsize=12)
#ylim(0, 90)
xlabel(r'$x \mathrm{\; (Unit. \, x)}$', fontsize=16)   
yticks(fontsize=12)
ylabel(r'$y \mathrm{\; (Unit. \, y)}$', fontsize=16)
#ylim(0, 90)

ax.legend(loc=4)
show()
