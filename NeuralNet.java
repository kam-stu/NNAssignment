import java.util.*;
import java.io.*;

class NeuralNet {

	// node amt for each layer
	private int inputLayer;
	private int hiddenLayer;
	private int outputLayer;

	// inputs
	private double[][] inputs;
	private double[][] outputs;

	private double[][][] miniBatchInputs;
	private double[][][] miniBatchOutputs;
	
	// outputs
	private double[][] Whidden;
	private double[] Bhidden;
	
	private double[][] Woutput;
	private double[] Boutput;

	private double learningRate;

	private boolean isTrained = false;
	private boolean hasWeights = false;
	
	public NeuralNet(int inputLayer, int hiddenLayer, int outputLayer, double learningRate) {
		this.inputLayer = inputLayer;
		this.hiddenLayer = hiddenLayer;
		this.outputLayer = outputLayer;
		this.learningRate = learningRate;
	}


	//  initializes weights and biases based on given values
	public void initWeights(double[][] W1, double[] B1, double[][] W2, double[] B2) {
		this.Whidden = W1;
		this.Bhidden = B1;
		this.Woutput = W2;
		this.Boutput = B2;

		this.hasWeights = true;
	}

	// parses CSV and adds inputs and outputs to their respective fields
	public void readCSV(String filename) {
		List<double[]> inputList = new ArrayList<>();
		List<double[]> outputList = new ArrayList<>();

		try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
			String line;
			while ((line=br.readLine()) != null) {
				String[] tokens = line.split(",");
				int label = Integer.parseInt(tokens[0]);
				
				// creates a one hot arr
				double[] output = new double[10];
				output[label] = 1;

				double[] input = new double[tokens.length -1];
				for (int i=1; i<tokens.length; i++) {
					// i-1 to make up for removing token[0] for output
					// tokens[i] / 255.0 naturalizes input (0.0 - 1.0)
					input[i-1] = Double.parseDouble(tokens[i]) / 255.0;
				}

				inputList.add(input);
				outputList.add(output);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}


		this.inputs = inputList.toArray(new double[inputList.size()][]);
		this.outputs = outputList.toArray(new double[outputList.size()][]);
	}

	private void initWeights() {
		// initialize sizes for all arrays
		this.Whidden = new double[this.hiddenLayer][this.inputLayer];
		this.Bhidden = new double[this.hiddenLayer];
		this.Woutput = new double[this.outputLayer][this.hiddenLayer];
		this.Boutput = new double[this.outputLayer];

		Random rand = new Random();
		
		// initialize random weights and biases
		// random values are in between -1 and 1
		for (int i=0; i<Whidden.length; i++) {
			for (int j=0; j<Whidden[i].length; j++) {
				Whidden[i][j] = rand.nextDouble() * 2 - 1;
			}
		}
		for (int i=0; i<Bhidden.length; i++) {
			Bhidden[i] = rand.nextDouble() * 2 - 1;
		}
		for (int i=0; i<Woutput.length; i++) {
			for (int j=0; j<Woutput[i].length; j++) {
				Woutput[i][j] = rand.nextDouble() * 2 - 1;
			}
		}
		for (int i=0; i<Boutput.length; i++) {
			Boutput[i] = rand.nextDouble() * 2 - 1;
		}

		this.hasWeights = true;
		
	}

	// implicitly generates a random batch based on total num of batches
	public void createMiniBatches(int batchSize) {
		int total = this.inputs.length;
		int numBatches = (int) Math.ceil((double) total / batchSize);

		int curr = 0;

		miniBatchInputs = new double[numBatches][][];
		miniBatchOutputs = new double[numBatches][][];

		for (int batch=0; batch<numBatches; batch++) {
			int currBatchSize = Math.min(batchSize, total - curr);
			miniBatchInputs[batch] = new double[currBatchSize][];
			miniBatchOutputs[batch] = new double[currBatchSize][];

			for (int i=0; i<currBatchSize; i++) {
				miniBatchInputs[batch][i] = inputs[curr];
				miniBatchOutputs[batch][i] = outputs[curr];
				curr++;
			}
		}
	}

	private void shuffleData() {
		int n = this.inputs.length;
		Random rand = new Random();

		for (int i=n-1; i>0; i--) {
			int j = rand.nextInt(i+1); // gets random index

			// swap inputs
			double[] tempInput = this.inputs[i];
			this.inputs[i] = this.inputs[j];
			this.inputs[j] = tempInput;

			// swap outputs
			double[] tempOut = this.outputs[i];
			this.outputs[i] = this.outputs[j];
			this.outputs[j] = tempOut;
		}
	}
	

	public void train(int epochs, int batchSize) {
		
		initWeights();
	
		for (int epoch=0; epoch<epochs; epoch++) {
			shuffleData();
			createMiniBatches(batchSize);
			
			// values for displaying correct vs seen
			// int[] represents the correct and total for number = index
			// ie: correct[0] shows all correct values for the digit 0
			int[] correctDigit = new int[10];
			int[] totalDigit = new int[10];
			int totalCorrect = 0;
			int totalSeen = 0;

			// loop through batches -> create sum arrays for sum of gradients
			for (int batch =0; batch < this.miniBatchInputs.length; batch++) {
				double[][] w1Sum = new double[Whidden.length][Whidden[0].length];
				double[] b1Sum = new double[Bhidden.length];
				double[][] w2Sum = new double[Woutput.length][Woutput[0].length];
				double[] b2Sum = new double[Boutput.length];

				// loops through the sets of a batch and forward prop
				for (int set=0; set<this.miniBatchInputs[batch].length; set++) {
					double[] A1 = forwardProp(this.miniBatchInputs[batch][set], Whidden, Bhidden);
					double[] A2 = forwardProp(A1, Woutput, Boutput);
					double[] b2G = backPropFinal(A2, this.miniBatchOutputs[batch][set]);
					double[][] w2G = prod(b2G, A1);
					setSum(w2Sum, w2G);
					setSum(b2Sum, b2G);

					double[] b1G = backPropHidden(b2G, A1);
					double[][] w1G = prod(b1G, this.miniBatchInputs[batch][set]);
					setSum(w1Sum, w1G);
					setSum(b1Sum, b1G);

					// get accuracy numbers
					int predicted = argMax(A2);
					int actual = argMax(this.miniBatchOutputs[batch][set]);
					totalSeen++;
					totalDigit[actual]++;
					if (predicted == actual) {
						correctDigit[actual]++;
						totalCorrect++;
					}

				}
				// batch finished -> update bias and weights
				updateB(b2Sum, Boutput, batchSize);
				updateW(w2Sum, Woutput, batchSize);
				updateB(b1Sum, Bhidden, batchSize);
				updateW(w1Sum, Whidden, batchSize);
			}
			System.out.println("Epoch " + (epoch+1) + " satistics: ");
			for (int i=0; i<10; i++) {
				System.out.print("Digit " + i +": " + correctDigit[i] + "/" + totalDigit[i] + "\t\t");
				if (i % 2 == 1) {
					System.out.println();
				}
			}
			double accuracy = 100.0 * totalCorrect / totalSeen;
			System.out.println("Accuracy: " + totalCorrect + "/" + totalSeen + " = " + accuracy + "%");
			System.out.println();
		}
		
		this.isTrained = true;
	}

	// tests a batch of inputs and outputs
	// assumes that the test inputs and outputs are set to the input and output
	// values for the NeuralNetwork
	public void testBatch() {
    		if (!isTrained) {
        		System.out.println("Network is not trained yet!");
        		return;
    	}

    		int totalCorrect = 0;
    		int totalSeen = inputs.length;
    		int[] correctDigit = new int[10];
    		int[] totalDigit = new int[10];

    		for (int i = 0; i < inputs.length; i++) {
        		double[] hiddenOut = forwardProp(inputs[i], Whidden, Bhidden);
        		double[] finalOut = forwardProp(hiddenOut, Woutput, Boutput);

        		int predicted = argMax(finalOut);
        		int actual = argMax(outputs[i]);

        		totalDigit[actual]++;
        		if (predicted == actual) {
            			correctDigit[actual]++;
            			totalCorrect++;
        		}
    		}

    		System.out.println("Batch Test Results:");
    		for (int i = 0; i < 10; i++) {
        		System.out.println("Digit " + i + ": " + correctDigit[i] + "/" + totalDigit[i]);
    		}

    		double accuracy = 100.0 * totalCorrect / totalSeen;
    		System.out.println("Overall Accuracy: " + totalCorrect + "/" + totalSeen + " = " + accuracy + "%");
	}


	// tests the network on a single given input
	public int test(double[] input) {
		if (!isTrained) {
			System.out.println("Network is not trained yet!");
			return -1;
		}

		asciiArt(input);

		double[] hiddenOut = forwardProp(input, Whidden, Bhidden);
		double[] finalOut = forwardProp(hiddenOut, Woutput, Boutput);

		return argMax(finalOut);
	}

	// function for getting the max activation 
	// returns the index of max value
	private int argMax(double[] activations) {
		int maxIndex = 0;
		double maxVal = activations[0];

		for (int i=0; i<activations.length; i++) {
			if (activations[i] > maxVal) {
				maxVal = activations[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	public void loadWeights() {}
	
	private double[] forwardProp(double[] A0, double[][] W, double[] B) {
		double[] A = new double[B.length];
		
		for (int i=0; i<B.length; i++) {
			double z = B[i];
			for (int j=0; j<W[i].length; j++) {
				z += W[i][j] * A0[j];
			}
			A[i] = sigmoid(z);
		}
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
			B[i] -= ((double) this.learningRate / batchSize) * bG[i];
		}
	}

	// updates Weights
	private void updateW(double[][] wG, double[][] W, int batchSize) {
		for (int i=0; i<W.length; i++) {
			for (int j=0; j<W[i].length; j++) {
				W[i][j] -= ((double) this.learningRate / batchSize) * wG[i][j];
			}
		}
	}

	private void asciiArt(double[] input) {
		for(int i=0; i< input.length; i++) {
			System.out.print(generateChar(input[i]));

			// because original image is 28x28
			// keeps the formatting correct
			if ((i+1)%28 == 0) {
				System.out.println();
			}
		}
	}
	
	// Holds the ranges for what each input should return (1.0 => pure white pixel => return #
	// Made this function then IMMEDIATELY realized i couldve just used a hashmap LOL
	private char generateChar(double value) {
		if (value <= 0.05) return ' ';
		if (value <= 0.2) return '.';
		if (value <= 0.5) return '!';
		if (value <= 0.10) return '*';
		if (value <= 0.15) return '^';
		if (value <= 0.25) return 'o';
		if (value <= 0.35) return '+';
		if (value <= 0.50) return '=';
		if (value <= 0.60) return '$';
		if (value <= 0.75) return '%';
		if (value <= 0.85) return '@';
		if (value <= 1.0) return '#';
		return '!';
	}

	private void printArr(double[] arr) {
		for (int i=0; i<arr.length; i++) {
			System.out.println(arr[i]);
		}
	}

	private void printMatrix(double[][] matrix) {
		for (int i=0; i<matrix.length; i++) {
			for (int j=0; j<matrix[i].length; j++) {
				System.out.print(matrix[i][j] + "\t");
			}
			System.out.println();
		}
	}
}
