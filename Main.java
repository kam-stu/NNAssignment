class Main {
	public static void main(String[] args) {
		NeuralNet nn = new NeuralNet(784, 15, 10, 3);

		nn.readCSV("mnist_train.csv");

		nn.train(30, 10);

		System.out.println("========================================");
		nn.readCSV("mnist_test.csv");
		nn.testBatch();
	}
}
