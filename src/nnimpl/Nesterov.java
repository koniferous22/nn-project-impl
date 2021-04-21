package nnimpl;

// SLIGHTLY MODIFIER IN COMPARISION TO FORMAL NESTEROV
public class Nesterov implements GradientDescent {

	private double learningRate;
	private double previousChange;

	private double gradientValue;

	private double [][][] changeBuffer; 

	public Nesterov(int[] networkInfo, double learningRate, double previousChange) {
		this.changeBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.learningRate = learningRate;
		this.previousChange = previousChange;
		this.gradientValue = 0.0;
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {
		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					gradientValue = epochBuffer[iLayer][iInputNeuron][iOuputNeuron];
					changeBuffer[iLayer][iInputNeuron][iOuputNeuron] *= previousChange;
					changeBuffer[iLayer][iInputNeuron][iOuputNeuron] += gradientValue * previousChange; 
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= previousChange * changeBuffer[iLayer][iInputNeuron][iOuputNeuron] + learningRate * gradientValue;
				}
			}
		}
	}
}