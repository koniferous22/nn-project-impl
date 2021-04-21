package nnimpl;

public class Momentum implements GradientDescent {

	private double learningRate;
	private double previousChange;

	private double [][][] changeBuffer; 

	public Momentum(int[] networkInfo, double learningRate, double previousChange) {
		this.changeBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.learningRate = learningRate;
		this.previousChange = previousChange;
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {
		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					changeBuffer[iLayer][iInputNeuron][iOuputNeuron] *= previousChange;
					changeBuffer[iLayer][iInputNeuron][iOuputNeuron	] += epochBuffer[iLayer][iInputNeuron][iOuputNeuron] * previousChange; 
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= changeBuffer[iLayer][iInputNeuron][iOuputNeuron];
				}
			}
		}
	}
}