package nnimpl;

import java.lang.Math;

public class HyperbolicTangent implements LayerActivator {
	private double a;
	private double b;

	public HyperbolicTangent(double a, double b) {
		this.a = a;
		this.b = b;
	}

	public HyperbolicTangent() {
		this(1.0, 1.0);
	}

	@Override
	public void activateLayer(double[] innerPotentials, double[] outputArray) {
		for (int i = 0 ; i < innerPotentials.length ; ++i) {
			double exponent = b * (-innerPotentials[i]);
			outputArray[i] = a * (1 - Math.exp(exponent)) / (1 + Math.exp(exponent));
		}
	}
	@Override
	public void derivateLayer(double[] neuronValues, double[] outputArray) {
		for (int i = 0 ; i < neuronValues.length ; ++i) {
			double value = neuronValues[i];
			outputArray[i] = (b / a) * (a - value) * (a + value);
		}
	}
}