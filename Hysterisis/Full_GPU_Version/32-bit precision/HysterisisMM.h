
void fillWeights(float *weights, int dim);

void fillLayers(float *weights, int dim);

void matrixMultiplication(float *layers, float *weights, float *external, int dim, int iterations, float timestep, float noise, float L, float M);