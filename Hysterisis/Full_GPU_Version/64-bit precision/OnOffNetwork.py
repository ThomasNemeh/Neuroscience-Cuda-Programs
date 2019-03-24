import HysterisisMM as hys
import numpy as np

V = np.array([0, 0])
W = np.array([[2, -1],
             [.5, 2]])

ext_input_1 = [0] * 50
ext_input_2 = [1.5]*50

ext_input = [ext_input_1, ext_input_2]


E=np.asarray(ext_input)

timestep = .1
noise = 0

LB = np.array([(6, .5), (6,1)])
#LB = [(6, .5), (6,1)]
#LB = (6, .5) #all neurons will have the same lambda/beta

#print("Timestep: " + str(timestep))

x = hys.hysterisis(200, timestep, noise, LB, layers=V, weights=W)
hys.printLastResults()
hys.writeResultToFile('OnOff', x)

'''
V = np.array([0])
W = np.array([2])
timestep = .1
noise = 0

#print("Timestep: " + str(timestep))

x = hys.hysterisis(200, timestep, noise, 6, .5, layers=V, weights=W)
hys.printLastResults()
'''
#When we add the inhibitory input, we need to make the self excitatory weight of the
#first neuron larger for it to turn on, meaning that we need a greater inhibitory input
#to turn it off.
