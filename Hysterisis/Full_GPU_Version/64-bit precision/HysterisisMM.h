
void fillWeights(double *weights, int dim);

void fillLayers(double *weights, int dim);

void matrixMultiplication(double *layers, double *weights, double *external, double *lamBeta, int dim, int iterations, float timestep, double noise);
