class Main {
	public static void main(String[] args) {
		NeuralNet nn = new NeuralNet(0,0,0);
		double[] a = new double[] {0.5,0.2};
		double[] b = new double[] {0.6,1.0};

		System.out.println(nn.forwardProp(a, b, 0));
	}
}
