package nnimpl;

import java.lang.Math;

public class RMSProp implements GradientDescent {

	private double lambda;
	private double learningRate;
	private double smoothingTerm;

	private double [][][] rmsPropBuffer;
	
	private double gradientValue;

	public RMSProp(
		int[] networkInfo,
		double lambda,
		double learningRate,
		double smoothingTerm
	) {
		this.rmsPropBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		
		this.lambda = lambda;
		this.learningRate = learningRate;
		this.smoothingTerm = smoothingTerm;

		this.gradientValue = 0.0;
}

	public RMSProp(int[] networkInfo, double lambda, double learningRate) {
		this(networkInfo, lambda, learningRate, Math.exp(-8) );
	}

	public RMSProp(int[] networkInfo, double lambda) {
		this(networkInfo, lambda, 0.01);
	}

	public RMSProp(int[] networkInfo) {
		this(networkInfo, 0.9);
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {

		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					gradientValue = epochBuffer[iLayer][iInputNeuron][iOuputNeuron];
					rmsPropBuffer[iLayer][iInputNeuron][iOuputNeuron] *= lambda;
					rmsPropBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - lambda) * gradientValue * gradientValue;
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= (learningRate * gradientValue) / Math.sqrt(rmsPropBuffer[iLayer][iInputNeuron][iOuputNeuron] + smoothingTerm);
				}
			}
		}
	}
}