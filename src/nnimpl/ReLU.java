package nnimpl;

import java.lang.Math;

public class ReLU implements LayerActivator {
	
	@Override
	public void activateLayer(double[] innerPotentials, double[] outputArray) {
		for (int i = 0 ; i < innerPotentials.length ; ++i) {
			outputArray[i] = Math.max(innerPotentials[i], 0.0001);
		}
	}
	
	@Override
	public void derivateLayer(double[] neuronValues, double[] outputArray) {
		for (int i = 0 ; i < neuronValues.length ; ++i) {
			outputArray[i] = (neuronValues[i] > 0) ? 1.0 : 0.0001;
		}
	}
}
