package nnimpl;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Project {
	private static String getPath(String relativePath) {
		Path currentPath = Paths.get(System.getProperty("user.dir"));
		Path filePath = Paths.get(currentPath.toString(), relativePath);
		return filePath.toString();
	}

	public static void main(String [] args) throws IOException {

		boolean mnist = true;

		// CONFIG
		// Rule of thumb: start with 10 times less neurons as size of training set
		int[] networkConfigXor = new int[]{
			2,	// FORMAL INPUT LAYER HERE
			100,
			100,
			1	// OUTPUT LAYER
		};

		int[] networkConfigMnist = new int[]{
			784,	// FORMAL INPUT LAYER HERE
			333,
			88,
			10	// OUTPUT LAYER LogisticSigmoid
		};

		double[] networkDropoutMnist = new double[]{
			0.0,
			0.0,
			0.0
		};

		int[] networkConfig = mnist ? networkConfigMnist : networkConfigXor;

		double[][] inputs = mnist ? 
			NetworkDataReader.loadMnistInputs(Project.getPath("data/MNIST/mnist_train_vectors.csv"), 60000, networkConfig[0]) : 
			NetworkDataReader.loadBasicStuff(Project.getPath("data/xor/xor-input.csv"), 4, networkConfig[0]);
		double[][] outputs = mnist ?
			NetworkDataReader.loadMnistOutputs(Project.getPath("data/MNIST/mnist_train_labels.csv"), 60000, networkConfig[networkConfig.length - 1]) :
			NetworkDataReader.loadBasicStuff(Project.getPath("data/xor/xor-output.csv"), 4, networkConfig[networkConfig.length - 1]);

		int epochs = mnist ? 10 : 1000;

		LayerActivator [] layerActivators = new LayerActivator[] {
			//new LogisticSigmoid(1.6),
			new LeakyReLU(0.001),
			new LeakyReLU(0.001),
			new Softmax()
			//new LogisticSigmoid(1.6)
		};

		// IF WANNA IMPLEMENT 1/p, implent it as Gradient Descent Constant
		ErrorFunction errorFunction = new CrossEntropy();
		GradientDescent gradientDescent = new RMSProp(networkConfig);
		WeightInitialization weightInitialization = new HeyThatsPrettyGoodWeightInitialization();

		FeedForwardNetwork ffn = new FeedForwardNetwork(
				networkConfig,
				layerActivators,
				errorFunction,
				gradientDescent,
				weightInitialization,
				networkDropoutMnist
			).learn(
				inputs,
				outputs,
				epochs,
				120
			);

		if (mnist) {
			inputs = NetworkDataReader.loadMnistInputs(Project.getPath("data/MNIST/mnist_train_vectors.csv"), 60000, networkConfig[0]);
			outputs = NetworkDataReader.loadMnistOutputs(Project.getPath("data/MNIST/mnist_train_labels.csv"), 60000, networkConfig[networkConfig.length - 1]);

			ffn.verify(inputs, outputs, Project.getPath("actualTrainPredictions"));

			inputs = NetworkDataReader.loadMnistInputs(Project.getPath("data/MNIST/mnist_test_vectors.csv"), 10000, networkConfig[0]);
			outputs = NetworkDataReader.loadMnistOutputs(Project.getPath("data/MNIST/mnist_test_labels.csv"), 10000, networkConfig[networkConfig.length - 1]);

			ffn.verify(inputs, outputs, Project.getPath("actualTestPredictions"));
		}

		
	}
}

