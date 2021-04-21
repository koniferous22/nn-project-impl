package nnimpl;

import java.lang.Math;

public class LogisticSigmoid implements LayerActivator {

	private double steepness;

	public LogisticSigmoid(double steepness) {
		this.steepness = steepness;
	}

	public LogisticSigmoid() {
		this(1.0);
	}

	@Override
	public void activateLayer(double[] innerPotentials, double[] outputArray) {
		// Wir nehmen an dass, die größe von den Felden 'innerPotentionalen' und 'produktionFeld' 'gleich ist
		for (int i = 0 ; i < innerPotentials.length ; ++i) {
			outputArray[i] = 1 / (1 + Math.exp((-innerPotentials[i]) * steepness));
		}
	}
	
	@Override
	public void derivateLayer(double[] neuronValues, double[] outputArray) {
		for (int i = 0 ; i < neuronValues.length ; ++i) {
			double value = neuronValues[i];
			outputArray[i] = steepness * value * (1 - value);
		}
	}
}