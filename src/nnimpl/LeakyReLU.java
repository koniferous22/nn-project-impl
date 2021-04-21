package nnimpl;

public class LeakyReLU implements LayerActivator {
	
	private double steepnessWhenInputIsNegative;

	public LeakyReLU(double steepnessWhenInputIsNegative) {
		this.steepnessWhenInputIsNegative = steepnessWhenInputIsNegative;
	}
	
	@Override
	public void activateLayer(double[] innerPotentials, double[] outputArray) {
		for (int i = 0 ; i < innerPotentials.length ; ++i) {
			outputArray[i] = innerPotentials[i];
			if (outputArray[i] < 0) {
				outputArray[i] *= steepnessWhenInputIsNegative;
			}
		}
	}
	@Override
	public void derivateLayer(double[] neuronValues, double[] outputArray) {
		for (int i = 0 ; i < neuronValues.length ; ++i) {
			outputArray[i] = (neuronValues[i] < 0) ? steepnessWhenInputIsNegative : 1.0;
		}
	}
}