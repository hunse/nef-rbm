
import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as tt

# sigma = 0.1
# sigma = 0.01
sigma = 0.001
tau_ref = 0.002
tau_rc = 0.02
alpha = 1
# bias = 1
beta = 1
amp = 1. / 65

def nlif(x):
    dtype = theano.config.floatX
    sigma = tt.cast(0.05, dtype=dtype)
    tau_ref = tt.cast(0.002, dtype=dtype)
    tau_rc = tt.cast(0.02, dtype=dtype)
    alpha = tt.cast(1, dtype=dtype)
    beta = tt.cast(1, dtype=dtype)
    amp = tt.cast(1. / 65, dtype=dtype)

    j = alpha * x + beta - 1
    # j = sigma * tt.log1p(tt.exp(j / sigma))
    j = sigma * tt.nnet.softplus(j / sigma)
    v = amp / (tau_ref + tau_rc * tt.log1p(1. / j))
    return tt.switch(j > 0, v, 0.0)

x = tt.vector()
nlif = theano.function([x], nlif(x))
softplus = theano.function([x], tt.nnet.softplus(x))

dtype = theano.config.floatX
x = np.linspace(-5, 5, 501).astype(dtype)
r = nlif(x)

plt.figure(1)
plt.clf()
plt.plot(x, r)

# x = np.linspace(-1000, -40, 101).astype(dtype)
plt.figure(2)
plt.clf()
plt.plot(x, softplus(x))
