class Main {
	public static void main(String[] args) {
		NeuralNet nn = new NeuralNet(4,3,2,10,false);
		
		double[][][] input = {
			{ {0,1,0,1}, {1,0,1,0} }, 
			{ {0,0,1,1}, {1,1,0,0} }
		};

		double[][][] output = {
			{{0,1},{1,0}},
			{{0,1},{1,0}}
		};
		
		double[][] W1 = {
			{-0.21, 0.72, -0.25, 1},
			{-0.94, -0.41, -0.47, 0.63},
			{0.15, 0.55, -0.49, -0.75}
		};

		double[] B1 = {0.1, -0.36, -0.31};
		
		double[][] W2 = {
			{0.76, 0.48, -0.73},
			{0.34, 0.89, -0.23}
		};

		double[] B2 = {0.16, -0.46};

		nn.initWeights(W1, B1, W2, B2);

		nn.train(input, output, 6, 2);
	}
}
