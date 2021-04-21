package nnimpl;
import java.lang.Math;

public class AdaMaxInfinity implements GradientDescent {

	private int epochCounter;

	private double beta1;
	private double beta2;
	private double learningRate;
	private double smoothingTerm;

	private double [][][] mBuffer;
	private double [][][] uBuffer;
	
	private double gradientValue;
	private double mBufferSumEstimation;
	private double uBufferSumEstimation;

	public AdaMaxInfinity(
		int[] networkInfo, 
		double beta1,
		double beta2,
		double learningRate,
		double smoothingTerm
	) {
		this.epochCounter = 0;
		
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.learningRate = learningRate;
		this.smoothingTerm = smoothingTerm;

		this.mBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.uBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.gradientValue = 0.0;
	}

	public AdaMaxInfinity(
		int[] networkInfo,
		double beta1,
		double beta2,
		double learningRate
	) {
		this(networkInfo, beta1, beta2, learningRate, (Math.exp(-8)));
	}

	public AdaMaxInfinity(
		int[] networkInfo,
		double beta1,
		double beta2
	) {
		this(networkInfo, beta1, beta2, 0.01);
	}

	public AdaMaxInfinity(
		int[] networkInfo,
		double beta1
	) {
		this(networkInfo, beta1, 0.999);
	}

	public AdaMaxInfinity(
		int[] networkInfo
	) {
		this(networkInfo, 0.9);
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {
		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					gradientValue = epochBuffer[iLayer][iInputNeuron][iOuputNeuron];
					
					mBuffer[iLayer][iInputNeuron][iOuputNeuron] *= beta1;
					mBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - beta1) * gradientValue;

					uBuffer[iLayer][iInputNeuron][iOuputNeuron] = Math.max(
						beta2 * uBuffer[iLayer][iInputNeuron][iOuputNeuron],
						Math.abs(gradientValue)
					);

					mBufferSumEstimation = mBuffer[iLayer][iInputNeuron][iOuputNeuron] / (1 - Math.pow(beta1, epochCounter));
					uBufferSumEstimation = uBuffer[iLayer][iInputNeuron][iOuputNeuron];
				
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= (learningRate * mBufferSumEstimation) / uBufferSumEstimation;
				}
			}
		}

		epochCounter++;
	}
}