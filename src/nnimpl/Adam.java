package nnimpl;
import java.lang.Math;
import java.io.IOException;

public class Adam implements GradientDescent {

	private int epochCounter;

	private double beta1;
	private double beta2;
	private double learningRate;
	private double smoothingTerm;

	private double [][][] meanBuffer;
	private double [][][] varianceBuffer;
	
	private double gradientValue;
	private double meanBufferSumEstimation;
	private double varianceBufferSumEstimation;

	public Adam(
		int[] networkInfo, 
		double beta1,
		double beta2,
		double learningRate,
		double smoothingTerm
	) {
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.learningRate = learningRate;
		this.smoothingTerm = smoothingTerm;
		
		this.epochCounter = 0;
		this.meanBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.varianceBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.gradientValue = 0.0;
	}

	public Adam(
		int[] networkInfo,
		double beta1,
		double beta2,
		double learningRate
	) {
		this(networkInfo, beta1, beta2, learningRate,  (Math.exp(-8)));
	}	

	public Adam(
		int[] networkInfo,
		double beta1,
		double beta2
	) {
		this(networkInfo, beta1, beta2,  0.01);
	}

	public Adam(
		int[] networkInfo,
		double beta1
	) {
		this(networkInfo, beta1,  0.999);
	}

	public Adam(
		int[] networkInfo
	) {
		this(networkInfo,  0.9);
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {
		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					gradientValue = epochBuffer[iLayer][iInputNeuron][iOuputNeuron];					

					meanBuffer[iLayer][iInputNeuron][iOuputNeuron] *= beta1;
					meanBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - beta1) * gradientValue;

					varianceBuffer[iLayer][iInputNeuron][iOuputNeuron] *= beta2;
					varianceBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - beta2) * gradientValue * gradientValue;

					meanBufferSumEstimation = meanBuffer[iLayer][iInputNeuron][iOuputNeuron] / (1 - Math.pow(beta1, epochCounter));

					varianceBufferSumEstimation = varianceBuffer[iLayer][iInputNeuron][iOuputNeuron] / (1 - Math.pow(beta2, epochCounter));
				
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= (learningRate * meanBufferSumEstimation) / Math.sqrt(varianceBufferSumEstimation + smoothingTerm);

				}
			}
		}

		epochCounter++;
	}

}
