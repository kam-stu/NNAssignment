import java.util.Random;

class NeuralNet {

	// node amt for each layer
	private int inputLayer;
	private int hiddenLayer;
	private int outputLayer;
	
	// outputs
	private double[][] Whidden;
	private double[] Bhidden;
	
	private double[][] Woutput;
	private double[] Boutput;

	private int learningRate;

	private boolean isMNIST = true;
	
	public NeuralNet(int inputLayer, int hiddenLayer, int outputLayer, int learningRate) {
		this.inputLayer = inputLayer;
		this.hiddenLayer = hiddenLayer;
		this.outputLayer = outputLayer;
		this.learningRate = learningRate;
	}

	public NeuralNet(int inputLayer, int hiddenLayer, int outputLayer, int learningRate, boolean isMNIST) {
		this.inputLayer = inputLayer;
		this.hiddenLayer = hiddenLayer;
		this.outputLayer = outputLayer;
		this.learningRate = learningRate;
		this.isMNIST = isMNIST;
	}

	public void initWeights(double[][] W1, double[] B1, double[][] W2, double[] B2) {
		this.Whidden = W1;
		this.Bhidden = B1;
		this.Woutput = W2;
		this.Boutput = B2;
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

	public void train(double[][][] batches, double[][][] output, int epochs, int batchSize) {
		
		for (int epoch=0; epoch<epochs; epoch++) {

			// loop through batches -> create sum arrays for sum of gradients
			for (int batch =0; batch < batches.length; batch++) {
				double[][] w1Sum = new double[Whidden.length][Whidden[0].length];
				double[] b1Sum = new double[Bhidden.length];
				double[][] w2Sum = new double[Woutput.length][Woutput[0].length];
				double[] b2Sum = new double[Boutput.length];

				// loops through the sets of a batch and forward prop
				for (int set=0; set<batches[batch].length; set++) {
					System.out.println("=====================================================");
					System.out.println("Batch " + (batch+1) + "\t Set: " + (set+1)); 
					double[] A1 = forwardProp(batches[batch][set], Whidden, Bhidden);
					double[] A2 = forwardProp(A1, Woutput, Boutput);
					double[] b2G = backPropFinal(A2, output[batch][set]);
					double[][] w2G = prod(b2G, A1);
					setSum(w2Sum, w2G);
					setSum(b2Sum, b2G);

					double[] b1G = backPropHidden(b2G, A1);
					double[][] w1G = prod(b1G, batches[batch][set]);
					setSum(w1Sum, w1G);
					setSum(b1Sum, b1G);
				}

				// batch finished -> update bias and weights
				updateB(b2Sum, Boutput, batchSize);
				updateW(w2Sum, Woutput, batchSize);
				updateB(b1Sum, Bhidden, batchSize);
				updateW(w1Sum, Whidden, batchSize);
				System.out.println("======================== BATCH DONE ===================");
				System.out.println("B1: ");
				printArr(Bhidden);
				System.out.println("W1: ");
				printMatrix(Whidden);
				System.out.println("B2: ");
				printArr(Boutput);
				System.out.println("W2: ");
				printMatrix(Woutput);
			}
		}
	}
	
	private double[] forwardProp(double[] A0, double[][] W, double[] B) {
		double[] A = new double[B.length];
		
		for (int i=0; i<B.length; i++) {
			double z = B[i];
			for (int j=0; j<W[i].length; j++) {
				z += W[i][j] * A0[j];
			}
			A[i] = sigmoid(z);
		}
		System.out.println("=============================FORWARD PROP=============================");
		printArr(A);
		return A;
	}
	
	private double sigmoid(double input) {
		return 1 / (1 + Math.exp(-(input)));
	}

	private double[] backPropHidden(double[] bG, double[] A) {
		double[] res = hadamard(
			dot(transpose(Woutput), bG),
			hadamard(A, subtract(ones(A.length), A))
		);
		return res;
	}

	private double[] backPropFinal(double[] A, double[] Y) {
		double[] res = hadamard(
			subtract(A, Y),
			hadamard(A, subtract(ones(A.length), A))
		);
		return res;
	}

	private double[][] transpose(double[][] matrix) {
		double[][] temp = new double[matrix[0].length][matrix.length];
		for (int i=0; i<matrix.length; i++) {
			for (int j=0; j<matrix[i].length;j++) {
				temp[j][i] = matrix[i][j];
			}
		}
		return temp;
	}

	private double[] hadamard(double[] A, double[] B){
		double[] res = new double[A.length];
		for (int i=0; i<A.length; i++) {
			res[i] = A[i] * B[i];
		}
		return res;
	}

	// subtracts 2 arrays (A - B)
	private double[] subtract(double[] A, double[] B) {
		double[] res = new double[A.length];
		for (int i=0; i<A.length; i++) {
			res[i] = A[i] - B[i];
		}
		return res;
	}

	// creates an array of all 1's 
	private double[] ones(int length) {
		double[] res = new double[length];
		for (int i=0; i<length; i++) {
			res[i] = 1;			
		}
		return res;
	}

	private double[] dot(double[][] A, double[] B) {
		double[] res = new double[A.length];
		for (int i=0; i<A.length; i++) {
			double sum =0;
			for (int j=0; j<A[0].length; j++) {
				sum += A[i][j] * B[j];
			}
			res[i] = sum;
		}
		return res;
	}

	private double[][] prod(double[] A, double[] B) {
		double[][] res = new double[A.length][B.length];
		for (int i=0; i<A.length; i++) {
			for (int j=0; j<B.length; j++) {
				res[i][j] = A[i] * B[j];
			}
		}
		return res;
	}

	// Sets the sum of 1 matrix into another (used for updating Weights using the sum of Weight Gradient)
	private void setSum(double[][] arr, double[][] gradient) {
		for (int i=0; i<arr.length; i++) {
			for (int j=0; j<arr[i].length; j++) {
				arr[i][j] += gradient[i][j];
			}
		}
	}

	// Sets the sum of 1 array into another (used for updating Bias using the sum of Bias Gradient)
	private void setSum(double[] arr, double[] gradient) {
		for (int i=0; i<arr.length; i++) {
				arr[i] += gradient[i];
		}
	}

	// updates Bias
	private void updateB(double[] bG, double[] B, int batchSize) {
		for (int i=0; i<B.length; i++) {
			B[i] -= (this.learningRate / batchSize) * bG[i];
		}
	}

	// updates Weights
	private void updateW(double[][] wG, double[][] W, int batchSize) {
		for (int i=0; i<W.length; i++) {
			for (int j=0; j<W[i].length; j++) {
				W[i][j] -= (this.learningRate / batchSize) * wG[i][j];
			}
		}
	}

	private void printArr(double[] arr) {
		for (int i=0; i<arr.length; i++) {
			System.out.println(arr[i]);
		}
	}
	private void printMatrix(double[][] matrix) {
		for (int i=0; i<matrix.length; i++) {
			for (int j=0; j<matrix[i].length; j++) {
				System.out.println(matrix[i][j]);
			}
		}
	}
}
