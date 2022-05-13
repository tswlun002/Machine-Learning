import  numpy as np
import numpy.random
QMatrix = np.zeros((12,12))
QMatrix[10,3]=12
n =QMatrix[10,:].max()
print(n)
r = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
t = np.array([2,3])
print(t+r[3])

l = [1,3,4,5,6,7,]
print(l[:len(l)-2])
