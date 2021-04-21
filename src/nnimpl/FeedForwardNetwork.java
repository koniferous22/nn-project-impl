package nnimpl;
import java.util.Random;
import java.lang.Double;
import java.io.PrintWriter;
import java.io.IOException;

public class FeedForwardNetwork {

	private Random random;

	// basic shit
	private int [] layerInfo;
	private double [] layerDropouts;
	private double [][] neurons;
	private double [][][] weights;
	
	// Buffers, in order to reduce number of allocations
	private double [][] innerPotentialsBuffer;
	private double [][] backPropagationNeuronBuffer;
	private double [][] temporaryBackpropagationBuffer;
	private double [][][] backPropagationWeightBuffer;

	private int [][] dropoutMaskBuffer;

	private int [] verificationResults;

	// training interface
	private LayerActivator[] functions;
	private ErrorFunction errorFunction;
	private GradientDescent gradientDescent;
	private WeightInitialization weightInitialization;

	// cached constands
	private int layerCountExcludingInput;

	// 1st integer=input space
	// remaining=number of neurons in layers
	public FeedForwardNetwork(
		int [] layerInfo,
		LayerActivator[] functions,
		ErrorFunction errorFunction,
		GradientDescent gradientDescent,
		WeightInitialization weightInitialization,
		double [] layerDropouts
	) {
		initialize(layerInfo, functions, errorFunction, gradientDescent, weightInitialization, layerDropouts);
	}

	public FeedForwardNetwork initialize(
		int [] layerInfo, 
		LayerActivator[] functions, 
		ErrorFunction errorFunction, 
		GradientDescent gradientDescent,
		WeightInitialization weightInitialization,
		double [] layerDropouts
	) {
		if (layerInfo == null || layerInfo.length <= 2) {
			throw new IllegalArgumentException("a");	
		}

		if (functions == null || functions.length + 1 != layerInfo.length) {
			throw new IllegalArgumentException("b");
		}
		if (gradientDescent == null) {
			throw new IllegalArgumentException("c");
		}
		if (layerDropouts == null || layerDropouts.length + 1 != layerInfo.length) {
			throw new IllegalArgumentException("d");
		}
		for (int i = 0 ; i < functions.length ; ++i) {
			if (functions[i] == null) {
				throw new IllegalArgumentException("e");
			}
		}

		this.layerInfo = layerInfo;
		this.layerDropouts = layerDropouts;
		this.functions = functions;
		this.errorFunction = errorFunction;
		this.gradientDescent = gradientDescent;

		// initialize basic shit
		neurons = NetworkBuilder.buildFeedForwardNetworkNeurons(layerInfo);
		weights = weightInitialization.initializeWeights(layerInfo);

		// initialize buffers
		innerPotentialsBuffer = NetworkBuilder.buildFeedForwardNetworkNeurons(layerInfo, false);
		temporaryBackpropagationBuffer = NetworkBuilder.buildFeedForwardNetworkNeurons(layerInfo, false);
		backPropagationNeuronBuffer = NetworkBuilder.buildFeedForwardNetworkNeurons(layerInfo, false);
		backPropagationWeightBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(layerInfo, false);
		dropoutMaskBuffer = NetworkBuilder.buildFeedForwardDropouts(layerInfo);

		// initialize cached constants
		layerCountExcludingInput = layerInfo.length - 1;

    	this.random = new Random();

		return this;
	}

	public FeedForwardNetwork feedInput(double [] input) {
		neurons[0] = input;
		for (int iLayer = 0 ; iLayer < layerCountExcludingInput ; ++iLayer) {
			
			double [] outputLayer = neurons[iLayer + 1];
			double [] inputLayer = neurons[iLayer];
			double [] potentials = innerPotentialsBuffer[iLayer];
			double [][] currentWeights = weights[iLayer];
			int [] currentDropouts = dropoutMaskBuffer[iLayer];
			LayerActivator layerFunction = functions[iLayer];

			for (int iNeuron = 0 ; iNeuron < outputLayer.length ; ++iNeuron) {
				potentials[iNeuron] = 0.0;
				for (int iInputNeuron = 0 ; iInputNeuron < inputLayer.length ; ++iInputNeuron) {
					potentials[iNeuron] += inputLayer[iInputNeuron] * currentWeights[iInputNeuron][iNeuron];
				}
				layerFunction.activateLayer(potentials, outputLayer);
				
				outputLayer[iNeuron] *= 1 - layerDropouts[iLayer];
			}
		}
		return this;
	}

	// DIFFERENT: computation in recurrent
	private FeedForwardNetwork feedTrainingInput(double [] input) {
		// LATER COMMENT INPUT, IF DATA MATCHES

		dropoutRandomize();
		neurons[0] = input;
		for (int iLayer = 0 ; iLayer < layerCountExcludingInput ; ++iLayer) {
			
			double [] outputLayer = neurons[iLayer + 1];
			double [] inputLayer = neurons[iLayer];
			double [] potentials = innerPotentialsBuffer[iLayer];
			double [][] currentWeights = weights[iLayer];
			int [] currentDropouts = dropoutMaskBuffer[iLayer];
			LayerActivator layerFunction = functions[iLayer];

			for (int iNeuron = 0 ; iNeuron < outputLayer.length ; ++iNeuron) {
				potentials[iNeuron] = 0.0;
				if (currentDropouts[iNeuron] == 0) {
					outputLayer[iNeuron] = 0.0;
					continue;
				}
				for (int iInputNeuron = 0 ; iInputNeuron < inputLayer.length ; ++iInputNeuron) {
					// COMPUTING POTENTIAL AS VECTOR DISTANCE
					potentials[iNeuron] += inputLayer[iInputNeuron] * currentWeights[iInputNeuron][iNeuron];
				}
				layerFunction.activateLayer(potentials, outputLayer);
				
				// VERIFICATION PROBLEM
				//outputLayer[iNeuron] *= 1 - layerDropouts[iLayer];
			}
		}
		return this;
	}

	private void resetBackPropBuffer() {
		for (int iLayer = 0 ; iLayer < backPropagationWeightBuffer.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < backPropagationWeightBuffer[iLayer].length ; ++iInputNeuron) {
				for (int iOutputNeuron = 0 ; iOutputNeuron < backPropagationWeightBuffer[iLayer][iInputNeuron].length ; ++iOutputNeuron) {
					backPropagationWeightBuffer[iLayer][iInputNeuron][iOutputNeuron] = 0.0;
				}
			}
		}
	}

	// weight initialization

	// EXTRA STUFF
	// early stopping
	// bagging


	public FeedForwardNetwork learn(
		double [][] inputDataset,
		double [][] outputDataset,
		int epochs,
		int batchSize
	) {
		for (int iEpoch = 0 ; iEpoch < epochs ; ++iEpoch) {
			System.out.println("Epoch " + (iEpoch + 1));
			epoch(inputDataset, outputDataset, batchSize);
		}

		return this;
	}

	private void epoch(double [][] inputDataset,double [][] outputDataset, int batchSize) {
		double error = 0.0;
		int counter = 0;
		int correctlyClassified = 0;

		shuffleIndices(inputDataset,outputDataset);		

		for (int iExample = 0 ; iExample < inputDataset.length ; ++iExample) {
			feedTrainingInput(inputDataset[iExample]);

			if (correctlyClassified(getOutput(), outputDataset[iExample])) {
				++correctlyClassified;
			}
			error += errorFunction.calculateError(getOutput(), outputDataset[iExample]);
			backpropagation(outputDataset[iExample]);
			counter++;
			if (counter % batchSize == 0 || counter == inputDataset.length) {
				//System.out.println("GRADIENT  " + counter);
				meanGradient(batchSize);
				gradientDescent();
			}
			/*if (counter % 1000 == 0) {
				System.out.println("Hello " + counter);
			}*/
		}
		System.out.format("Loss: %.8f\nAccuracy = %.2f%% (%d/%d)\n", error / counter, 100.0 * correctlyClassified / counter, correctlyClassified, counter);
	}

	public double verify(
		double [][] inputDataset,
		double [][] outputDataset
	) {
		System.out.println("VERIFYING\n");

		double error = 0.0;
		int counter = 1;
		double errorExample = 0.0;
		int correctlyClassified = 0;

		verificationResults = new int[outputDataset.length];

		dropoutZero();

		for (int iExample = 0 ; iExample < inputDataset.length ; ++iExample) {
			feedInput(inputDataset[iExample]);
			verificationResults[iExample] = classify(getOutput());
			// comment this
			if (correctlyClassified(getOutput(), outputDataset[iExample])) {
				++correctlyClassified;
			}
			errorExample = errorFunction.calculateError(getOutput(), outputDataset[iExample]);
			error += errorExample;
			++counter;
		}

		System.out.format("Accuracy = %.2f%% (%d/%d)\n", 100.0 * correctlyClassified / counter, correctlyClassified, counter);
		return error / counter;
	}

	public double verify(
		double [][] inputDataset,
		double [][] outputDataset,
		String exportOutputPath
	) {
		double result = verify(inputDataset,outputDataset);
		try ( PrintWriter pw = new PrintWriter(exportOutputPath) ) {
			for (int i = 0 ; i < verificationResults.length ; ++i) {
				pw.println(verificationResults[i]);
			}
		} catch(IOException e) {
			System.out.println("Export failed");
		}
		

		return result;
	}

	public double[] getOutput() {
		return neurons[layerCountExcludingInput];
	}

	// IS THIS COMMUTATIVE???
	private void backpropagation(double [] expectedOutput) {
		// returns data for one input insertec into gradient descent buffer
		errorFunction.derivationOutputLayer(getOutput(), expectedOutput, backPropagationNeuronBuffer[layerCountExcludingInput - 1]);

		// iLayer is index for back propagation buffer: n-2 iterations (for every hidden layer)

		for (int iLayer = layerCountExcludingInput - 1; iLayer > 0 ; --iLayer) {
			backPropagationHiddenLayer(iLayer);
			// Here derivation of MSE by neurons is computed and saved to backPropagationNeuronBuffer

		}

		// construct gradient vector for following example -> backPropagationWeightBuffer
		for (int iLayer = layerCountExcludingInput - 1; iLayer >= 0 ; --iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weights[iLayer].length ; ++iInputNeuron) {
				for (int iOutputNeuron = 0 ; iOutputNeuron < weights[iLayer][iInputNeuron].length ; ++iOutputNeuron) {
 					backPropagationWeightBuffer[iLayer][iInputNeuron][iOutputNeuron] += backPropagationNeuronBuffer[iLayer][iOutputNeuron] * temporaryBackpropagationBuffer[iLayer][iOutputNeuron] * neurons[iLayer][iInputNeuron];
				}
			}
		}
	}
	
	private void backPropagationHiddenLayer(int hiddenLayerIndex) {
		functions[hiddenLayerIndex].derivateLayer(neurons[hiddenLayerIndex + 1], temporaryBackpropagationBuffer[hiddenLayerIndex]);

		for (int iNeuron = 0 ; iNeuron < layerInfo[hiddenLayerIndex] ; ++iNeuron) {
			backPropagationNeuronBuffer[hiddenLayerIndex - 1][iNeuron] = 0.0;
			if (dropoutMaskBuffer[hiddenLayerIndex - 1][iNeuron] == 0) {
				continue;
			}
			for (int iNeuronAbove = 0 ; iNeuronAbove < layerInfo[hiddenLayerIndex + 1] ; ++iNeuronAbove) {
				backPropagationNeuronBuffer[hiddenLayerIndex - 1][iNeuron] += 
					backPropagationNeuronBuffer[hiddenLayerIndex][iNeuronAbove] * 
					temporaryBackpropagationBuffer[hiddenLayerIndex][iNeuronAbove] * 
					weights[hiddenLayerIndex][iNeuron][iNeuronAbove];
			}
		}
	}

	private void gradientDescent() {
		gradientDescent.gradientDescent(weights, backPropagationWeightBuffer);
		resetBackPropBuffer();
	}

	private void meanGradient(int batchSize) {
        for (int i = 0; i < backPropagationWeightBuffer.length; i++) {
            for (int j = 0; j < backPropagationWeightBuffer[i].length; j++) {
                for (int k = 0; k < backPropagationWeightBuffer[i][j].length; k++) {
                    this.backPropagationWeightBuffer[i][j][k] /= batchSize;
                }
            }
        }
    }

	private void printNeurons() {
		for (int iLayer = layerCountExcludingInput ; iLayer >= 0 ; --iLayer) {
			System.out.println("LAYER " + iLayer);
			printLayer(iLayer);
			System.out.println();
		}
	}

	private void printLayer(int iLayer) {
		for (int iNeuron = 0 ; iNeuron < neurons[iLayer].length ; ++iNeuron) {
			System.out.format("%.2f ", neurons[iLayer][iNeuron]);
		}
	}

	private void printWeights() {
		for (int iLayer = layerCountExcludingInput - 1; iLayer >= 0 ; --iLayer) {
			System.out.println("LAYER " + iLayer + " -> LAYER " + (iLayer + 1));
			for (int iInputNeuron = 0 ; iInputNeuron < weights[iLayer].length ; ++iInputNeuron) {
				for (int iOutputNeuron = 0 ; iOutputNeuron < weights[iLayer][iInputNeuron].length ; ++iOutputNeuron) {
					System.out.print("\t" + iInputNeuron + " -> " + iOutputNeuron + ": ");
					System.out.format("%.2f\n", weights[iLayer][iInputNeuron][iOutputNeuron]);
				}
			}
			System.out.println("=====");
		}
	}

	private int classify(double [] arr) {
		int maxIndex = 0;
		for ( int i = 1; i < arr.length; i++ )
		{
			if ( arr[i] > arr[maxIndex] ) {
				maxIndex = i;
			}
		}
		return maxIndex;
	}

	private boolean correctlyClassified(double [] realOutput, double [] expectedOutput) {
		int classReal = classify(realOutput);
		int classExpected = classify(expectedOutput);
		return classReal == classExpected && (!Double.isNaN(realOutput[classReal]));
	}

	private void shuffleIndices(double[][] inputs, double[][] outputs) {
        int inputsLength = inputs.length;
        for (int i = inputsLength - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);

            double[] a = inputs[index];
            double[] b = outputs[index];
            inputs[index] = inputs[i];
            outputs[index] = outputs[i];
            inputs[i] = a;
            outputs[i] = b;
        }
    }

    private void dropoutRandomize() {
		for (int iLayer = 0 ; iLayer < dropoutMaskBuffer.length ; ++iLayer) {
			for (int iNeuron = 0 ; iNeuron < dropoutMaskBuffer[iLayer].length ; ++iNeuron) {
				dropoutMaskBuffer[iLayer][iNeuron] = (random.nextDouble() >= layerDropouts[iLayer]) ? 1 : 0;
			}
		}
    }
    private void dropoutZero() {
		for (int iLayer = 0 ; iLayer < dropoutMaskBuffer.length ; ++iLayer) {
			for (int iNeuron = 0 ; iNeuron < dropoutMaskBuffer[iLayer].length ; ++iNeuron) {
				dropoutMaskBuffer[iLayer][iNeuron] = 0;		
			}
		}
    }
}
