import HysterisisMM as hys
import numpy as np

V = np.array([0, 0, 0, 0])

#W = np.array([[0, -1, 1, 0],
#			 [-1, 0, 0, 1],
#			 [0, 0, 2, 0],
#			 [0, 0, 0, 2]])
'''W = np.array([[0, -1, 0, 0],
			 [-1, 0, 0, 0],
			 [1, 0, 2, 0],
			 [0, 1, 0, 2]])'''
W = np.array([[0, 0, 0, 0],
			 [0, 0, 0, 0],
			 [1000, 0, 0, 0],
			 [0, 2000, 0, 0]])


ext_input = [0] * 50
ext_input.extend([1.5]*50)
ext_input.extend([0]*300)



E=np.asarray(ext_input)

timestep = .1
noise = 0

#print("Timestep: " + str(timestep))

x = hys.hysterisis(100, timestep, noise, 4, 1.5, layers=V, weights=W, external=E)
#x = hys.hysterisis(100, timestep, noise, 4, 1.5, layers=V, weights=W)
hys.printLastResults()
#print(x)
hys.writeResultToFile('FourNeurons', x)


'''
V = np.array([0])
W = np.array([2])
timestep = .1
noise = 0

ext_input = [0] * 50
ext_input.extend([-1]*50)

E=np.asarray(ext_input)

#print("Timestep: " + str(timestep))

x = hys.hysterisis(100, timestep, noise, 4, .5, layers=V, weights=W, external=E)
hys.printLastResults()
'''
