import HysterisisMM as hys
import numpy as np

V = np.array([.05])
W = np.array([.1])
timestep = .1
noise = 0

#print("Timestep: " + str(timestep))

x = hys.hysterisis(100, timestep, noise, 4, .5, layers=V, weights=W)
hys.printLastResults()
'''
V = np.array([0])
W = np.array([2])
timestep = .1
noise = 0

#print("Timestep: " + str(timestep))

x = hys.hysterisis(100, timestep, noise, 10, .5, layers=V, weights=W)
hys.printLastResults()

V = np.array([.499])
W = np.array([1])
timestep = 1
noise = 0

#print("Timestep: " + str(timestep))

x = hys.hysterisis(100, timestep, noise, 20, .5, layers=V, weights=W)
hys.printLastResults()


V = np.array([.0605])
W = np.array([1.5])
timestep = 1
noise = 0

#print("Timestep: " + str(timestep))

x = hys.hysterisis(100, timestep, noise, 20, .5, layers=V, weights=W)
hys.printLastResults()

#******************************************************************

V = np.array([.6,.6,.1])

W = np.array([[1, 0, 0],
			 [0, 1, 0],
			 [.25, .25, 1]])

timestep = .1
noise = .00

x = hys.hysterisis(100, timestep, noise, 20, .5, layers=V, weights=W)
hys.printLastResults()

'''
