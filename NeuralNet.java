import java.util.Random;

class NeuralNet {

	// learning and testing dataset - stored as CSV's
	private String learningCSV;
	private String testingCSV;

	// node amt for each layer
	private int inputLayer;
	private int hiddenLayer;
	private int outputLayer;

	// arrays for the mini batches
	private double[][] batchInput;
	private double[][] batchOutput;
	
	// outputs
	private double[][] Whidden;
	private double[] Bhidden;
	
	private double[][] Woutput;
	private double[] Boutput;

	private int learningRate;
	
	public NeuralNet(int inputLayer, int hiddenLayer, int outputLayer) {
		this.inputLayer = inputLayer;
		this.hiddenLayer = hiddenLayer;
		this.outputLayer = outputLayer;
	}

	private void train(double[][] inputTrain, double[][] outputTrain, int epoch) {
		// put input into mini batch

		for (int i=0; i<epoch; i++) {
			// loop through mini batches
				// forward prop
				// compute loss
				// backprop
		}
		
	}

	private void initWeights() {
		// initialize sizes for all arrays
		this.Whidden = new double[this.inputLayer][this.hiddenLayer];
		this.Bhidden = new double[this.hiddenLayer];
		this.Woutput = new double[this.hiddenLayer][this.outputLayer];
		this.Boutput = new double[this.outputLayer];

		Random rand = new Random();
		
		// initialize random weights and biases
		for (int i=0; i<Whidden.length; i++) {
			for (int j=0; j<Whidden[i].length; j++) {
				Whidden[i][j] = rand.nextDouble() * 0.1;
			}
		}
		for (int i=0; i<Bhidden.length; i++) {
			Bhidden[i] = 0;
		}
		for (int i=0; i<Woutput.length; i++) {
			for (int j=0; j<Woutput[i].length; j++) {
				Woutput[i][j] = rand.nextDouble() * 0.1;
			}
		}
		for (int i=0; i<Boutput.length; i++) {
			Boutput[i] = 1;
		}
		
	}

	private double[] forwardProp(double[] X, double[][] W, double[] b) {
		double[] activations = new double[b.length];
		
		for (int i=0; i<b.length; i++) {
			double Z = 0.0;
			for (int j=0; j<X.length; j++) {
				Z += X[j] * W[j][i]; // dot product
			}
			Z += b[i];
			activations[i] = this.sigmoid(Z);
		}

		return activations;
	}

	private double sigmoid(double input) {
		return 1 / (1 + Math.exp(-(input)));
	}
	
	private void backProp(double[][] X, double[][] Y, double[][]A1, double[][]A2) {
		int m = X.length;
		
		// calculating gradients
		double[][] B2 = hadamard(
				hadamard(subtract(A2, Y), A2),
				subtract(ones(A2.length, A2[0].length), A2)
				);
		double[][] W2 = dot(B2, transpose(A1));

		double[][] B1 = hadamard(
				dot(transpose(W2), B2),
				hadamard(A1, hadamard(ones(A1.length, A1[0].length), A1))
				);

		double[][] W1 = dot(B1, transpose(X));

		// updating hidden weight
		for (int i=0; i<Whidden.length; i++) {
			for (int j=0; j<Whidden[i].length; j++) {
				Whidden[i][j] -= this.learningRate / m * W1[i][j];
			}
		}

		// update output weight
		for (int i=0; i<Woutput.length; i++) {
			for (int j=0; j<Woutput[i].length; j++) {
				Woutput[i][j] -= this.learningRate / m * W2[i][j];
			}
		}

		// update hidden bias
		for (int i=0; i<Bhidden.length; i++) {
			double sum =0.0;
			for (int j=0; j<B1.length; j++) {
				sum += B1[j][i];
			}
			Bhidden[i] -= learningRate / m * sum;
		}

		// update output bias
		for (int i=0; i<Boutput.length; i++) {
			double sum =0.0;
			for (int j=0; j<B2.length; j++) {
				sum += B2[j][i];
			}
			Boutput[i] -= learningRate / m * sum;
		}

	}

	private double[][] parser(String file) {return null;}

	private double[][] transpose(double[][] matrix) {
		double[][] temp = new double[matrix[0].length][matrix.length];
		for (int i=0; i<matrix.length; i++) {
			for (int j=0; j<matrix[i].length;j++) {
				temp[j][i] = matrix[i][j];
			}
		}
		return temp;
	}
	private double[][] hadamard(double[][] A, double[][] B) {
		double[][] res = new double[A.length][A[0].length];
		
		for (int i=0; i<A.length; i++) {
			for (int j=0; j<A[0].length; j++) {
				res[i][j] = A[i][j] * B[i][j];
			}
		}
		return res;
	}
	private double[][] subtract(double[][] A, double[][] B) {
		double[][] res = new double[A.length][A[0].length];
		for (int i=0; i<A.length; i++) {
			for (int j=0; j<A[i].length; j++) {
				res[i][j] = A[i][j] - B[i][j];
			}
		}
		return res;
	}
	private double[][] ones(int rows, int cols) {
		double[][] res = new double[rows][cols];
		for (int i=0; i<rows; i++) {
			for (int j=0; j<cols; j++) {
				res[i][j] = 1.0;
			}
		}
		return res;
	}
	private double[][] dot(double[][] A, double[][] B) {
		int rowsA = A.length;
		int colsA = A[0].length;
		int rowsB = B.length;
		int colsB = B[0].length;

		double[][] res = new double[rowsA][colsB];

		for (int i=0; i<rowsA; i++) {
			for (int j=0; j<colsB; j++) {
				double sum =0.0;
				for (int k=0; k<colsA; k++) {
					sum+= A[i][k] * B[k][j];
				}
				res[i][j] = sum;
			}
		}
		return res;
	}
}
