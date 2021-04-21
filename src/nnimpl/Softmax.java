package nnimpl;

import java.lang.Math;

public class Softmax implements LayerActivator {

	@Override
	public void activateLayer(double[] innerPotentials, double[] outputArray) {
		double max = Double.NEGATIVE_INFINITY;

		for (int i = 0 ; i < innerPotentials.length ; ++i) {
			max = Math.max(innerPotentials[i], max);
		}
		double menovathel = 0.0;
		for (int i = 0 ; i < innerPotentials.length ; ++i) {
			menovathel += Math.exp(innerPotentials[i] - max);
		} 
		for (int i = 0 ; i < innerPotentials.length ; ++i) {
			outputArray[i] = Math.exp(innerPotentials[i] - max) / menovathel;
		}
	}
	
	@Override
	public void derivateLayer(double[] neuronValues, double[] outputArray) {
		for (int i = 0 ; i < neuronValues.length ; ++i) {
			outputArray[i] = neuronValues[i] * (1 - neuronValues[i]);
		}
	}
}