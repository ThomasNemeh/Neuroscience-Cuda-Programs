To use my program, navigate to the github, then go to Hysterisis_Cuda_Programs/Updated_MM

To use the program, type:
> import HysterisisMM as hys 
in the python interpreter. 

Doing so will give you access to the following function:  
hysterisis(int iterations, float dt, float noise, float L, float B, np.ndarray layers = None, np.ndarray weights = None, np.ndarray external = None, int num_neurons = -1)

PARAMETERS:
	iterations: the number of cycles of processing the neural network. Must be a non-decimal number.
	
	dt: the timestep of the computation. 
	
	noise: parameter to calculate guassian noise: noise * N(0,1) * sqrt(timestep). dt should be a small value to be sensible, such as .1
		
	L: lam parameter for squeeze function
	
	B: beta paramter for squeeze function
		Squeeze function: 1 / (1 + expf(-1 * lam * (input - beta)));
		
	layers: Optional parameter. List of initial activation values for each neuron in the network. If not provided, the layers vector will be filled with random values between 0 and 1.
		layers should be a flat numpy vector. 

	weights: Optional paramter. List of weight connections between neurons. If not provided, the weights matrix will be filled with random integer values between 0 and 10.
		If the size of layers is n, weights can be an n x n matrix or a single flat vector of size n ^2.
	
	external: Optional parameter. External inputs to each neuron at each iteration. If not provided no external inputs will be involved in the computation. If the size of layer is 
		n, the size of external should be n * iterations * num_neurons. It can be a matrix or a flat vector, but should be the full size to produce the intended results.
	
	num_neurons: Optional parameter. Number of neurons in the network. Necessary if layers and weights are not provided. 
	
See HysterisisMMTest.py for some simple examples of using this function.

OTHER USEFUL FUNCTIONS:
	printLastResults(): print results for each neuron column by column from last call of hysterisis()
	
	printResults(array, length, iterations): Same as above, but with the given parameters. 
	
	writeResultToFile(fileName): Write results array from last call of hysterisis() into file of given name. 
	
    writeResultToFile(fileName, fileName): Same as above, but with the given array
	
	readFromFile(fileName): returns array in given flie.
    
	
	


	
	

