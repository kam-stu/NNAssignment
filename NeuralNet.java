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
	
	private void backProp(double[][] X, double[][] Y, double[][]Ai, double[][]Ao) {}

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
	private double[][] hadamardProd(double[][] A, double[][] B) {
		double[][] res = new double[A.length][A[0].length];
		
		for (int i=0; i<A.length; i++) {
			for (int j=0; j<A[0].length; j++) {
				res[i][j] = A[i][j] * B[i][j];
			}
		}
		return res;
	}
}
