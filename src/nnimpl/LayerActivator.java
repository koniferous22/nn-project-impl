package nnimpl;

public interface LayerActivator {
	public void activateLayer(double[] innerPotentials, double[] outputArray);
	public void derivateLayer(double[] neuronValue, double[] outputArray);
}
