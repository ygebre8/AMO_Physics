if True:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    

def Fun(eps, q = 1):
    bgr = 1/(1+q*q)
    res = np.power((q+eps),2.0)/(1+eps*eps)
    return bgr * res

eps = np.linspace(-20, 20, 500)
# plt.plot(eps, Fun(eps))
plt.plot(eps, Fun(eps, 1.5))
plt.title("Fano-lineshape")

plt.xlabel(r'Reduced Energy $\epsilon$ ')
plt.ylabel(r'$|1-S|^2$')

plt.axvline(x= 1.5 , color='r')

# plt.plot(eps, Fun(eps, 2.5))

plt.savefig("pic.png")