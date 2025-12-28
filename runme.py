from rawfile import rawread
from runtest import *
import numpy as np
import sys

tests=[]

tran1 = rawread('tran1.raw').get()
t = tran1["time"]
v1 = tran1["1"]
print("Points", t.size)

# sys.exit(0)

import matplotlib.pyplot as plt 

fig1, ax1 = plt.subplots(1, 1)
fig1.axes[0].set_title('Ring oscillator')
fig1.axes[0].set_ylabel('V [V]')
fig1.axes[0].set_xlabel('time [us]')
fig1.axes[0].plot(t*1e6, v1, color="blue", marker=".", label="v(1)")
plt.show()

