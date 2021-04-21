package nnimpl;

public class StochasticGradientDescent implements GradientDescent {
	private double learningRate;

	public StochasticGradientDescent(double learningRate) {
		this.learningRate = learningRate;
	}

	public StochasticGradientDescent() {
		this(0.1);
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {
		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= epochBuffer[iLayer][iInputNeuron][iOuputNeuron] * learningRate;
				}
			}
		}
	}
}
