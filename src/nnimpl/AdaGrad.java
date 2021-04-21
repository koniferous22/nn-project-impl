package nnimpl;
import java.lang.Math;

// TRY ADAGRAD ONLINE
public class AdaGrad implements GradientDescent {

	private double learningRate;
	private double smoothingTerm;

	private double [][][] sumOfSquaresOfPreviousGradients;
	
	private double gradientValue;

	public AdaGrad(int[] networkInfo, double learningRate, double smoothingTerm) {
		this.sumOfSquaresOfPreviousGradients = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.learningRate = learningRate;
		this.smoothingTerm = smoothingTerm;
		this.gradientValue = 0.0;
		this.learningRate = 0.0;
	}

	public AdaGrad(int[] networkInfo, double learningRate) {
		this(networkInfo, learningRate,  Math.exp(-8));
	}	

	public AdaGrad(int[] networkInfo) {
		this(networkInfo,  0.01);
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {
		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					gradientValue = epochBuffer[iLayer][iInputNeuron][iOuputNeuron];
					sumOfSquaresOfPreviousGradients[iLayer][iInputNeuron][iOuputNeuron] += gradientValue * gradientValue;
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= (learningRate * gradientValue) / Math.sqrt(sumOfSquaresOfPreviousGradients[iLayer][iInputNeuron][iOuputNeuron] + smoothingTerm);
				}
			}
		}
	}

}