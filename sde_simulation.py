import numpy as np
import sdeint
import matplotlib.pyplot as plt

a, l = 0.01, 3.835
sigma0, sigma1 = 0.02, 0.01

def f(x, t):
  mu0 = a*x[0] - 3.0/(8*l**2)*(2*x[0]*x[1]**2 + x[0]**3)
  mu1 = a*x[1] - 3.0/(8*l**2)*(2*x[0]**2*x[1] + x[1]**3)

  return np.array([mu0, mu1])

def g(x, t):
  sig0 = sigma0*x[0]
  sig1 = sigma1*x[1]

  return np.array([sig0, sig1]).reshape((2,1))

tspan = np.linspace(0.0, 5000.0, 500001)
x0 = np.array([0.05, 0.025])

x = sdeint.itoint(f, g, x0, tspan)
a
plt.plot(x[:,0], x[:,1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

np.savetxt('series.out', np.c_[tspan, x], delimiter=',')
