import numpy as np
import matplotlib.pyplot as plt

new = np.load('/home/kyle/iuvs/lut-16streams-lambert.npy')
old = np.load('/home/kyle/iuvs/lut-16streams-lambert-10x.npy')
throw = np.load('/home/kyle/iuvs/lut-throwback.npy')

foo = throw[:, :, :, 0]/new

print(np.mean(foo))

'''w = [1, 2, 3, 13, 14, 15]

foo = old/new

print(np.amax(foo), np.amin(foo), np.mean(foo))


fig, ax = plt.subplots()
for i in range(21):
    ax.plot(w, new[3, 3, 3, 5, i, 0, :])
    ax.plot(w, old[3, 3, 3, 5, i, 0, :])

plt.savefig('/home/kyle/iuvs/dustlut.png')'''
