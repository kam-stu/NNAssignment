import java.util.Scanner;

class Main {
	public static void main(String[] args) {
		boolean running = true;
		Scanner scanner = new Scanner(System.in);
		NeuralNet nn = new NeuralNet(784, 15, 10, 3);

		while (running) {
			System.out.println("1) Train the network \n2) Load a pre-trained network\n3) Display network accuracy on training data\n4) Display network accuracy on testing data\n5) Run network on testing data showing images and labels\n6) Display the misclassified testing images\n7) Save the network state to file\n0) Exit");
			String userInput = scanner.nextLine();
			
			// Check user input and call its expected function
			switch (userInput) {
				case "0":
					scanner.close();
					return;
				case "1":
					nn.clearTerminal();
					nn.train(30, 10);
					break;
				case "2":
					nn.clearTerminal();
					nn.loadWeights();
					break;
				case "3":
					nn.clearTerminal();
					nn.testBatch(false);
					break;
				case "4":
					nn.clearTerminal();
					nn.testBatch(true);
					break;
				case "5":
					nn.clearTerminal();
					nn.displayImage(false);
					break;
				case "6":
					nn.clearTerminal();
					nn.displayImage(true);
					break;
				case "7":
					nn.clearTerminal();
					nn.saveWeights();
					break;
				default: 
					nn.clearTerminal();
					continue;
			}
			
		}
	}
}
