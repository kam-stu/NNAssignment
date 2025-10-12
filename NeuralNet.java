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
	
	public NeuralNet(int inputLayer, int hiddenLayer, int outputLayer) {
		this.inputLayer = inputLayer;
		this.hiddenLayer = hiddenLayer;
		this.outputLayer = outputLayer;
	}

	private void train(double[][] inputTrain, double[][] outputTrain, int epoch) {
		for (int i=0; i<epoch; i++) {
			for (int j=0; j < inputTrain.length; j++) {
				double[] hiddenOut = forwardProp(inputTrain[j], Whidden, Bhidden);
				double[] finalOut = forwardProp(hiddenOut, Woutput, Boutput);

				// compute loss
				
				// backprop
			}
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
		double[] outputs = new double[b.length];
		
		for (int i=0; i<b.length; i++) {
			double Z = 0.0;
			for (int j=0; j<X.length; j++) {
				Z += X[j] * W[j][i]; // dot product
			}
			Z += b[i];
			outputs[i] = this.sigmoid(Z);
		}

		return outputs;
	}

	private double sigmoid(double input) {
		return 1 / (1 + Math.exp(-(input)));
	}
	
	private void backProp(double[] X, double[] Y) {}
}
