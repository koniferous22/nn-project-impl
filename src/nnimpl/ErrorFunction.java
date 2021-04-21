package nnimpl;
public interface ErrorFunction {
	public void derivationOutputLayer(
		double [] realOutput,
		double [] expectedOutput,
		double [] derivationOutput
	);

	public double calculateError(
		double [] realOutput,
		double [] expectedOutput
	);
}
