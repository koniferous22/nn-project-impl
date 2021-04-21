package nnimpl;

import java.lang.Math;

public class MeanSquareError implements ErrorFunction {
	@Override
	public void derivationOutputLayer(
		double [] realOutput, 
		double [] expectedOutput,
		double [] derivationOutput
	) {
		/* uncomment for bugs
		if (realOutput == null || expectedOutput == null || realOutput.length != expectedOutput.length) {
			throw new IllegalStateException("");
		}*/
		for (int i = 0 ; i < realOutput.length ; ++i) {
			derivationOutput[i] = realOutput[i] - expectedOutput[i];
		}
	}
	/*
	public void derivationHiddenLayer(
		LayerActivator activatorAbove,
		double [] derivationsAbove,
		double [] outputsAbove,
		double [][] weightsBetween,
		double [] temporaryBuffer,
		double [] derivationOutput
	) {
		activatorAbove.derivate(outputsAbove, temporaryBuffer);

		for (int iNeuron = 0 ; iNeuron < output.length ; ++iNeuron) {
			for (int iNeuronAbove = 0 ; iNeuronAbove < outputsAbove.length ; ++iNeuronAbove) {
				derivationOutput[iNeuron] += derivationsAbove[iNeuronAbove] * temporaryBuffer[iNeuronAbove] * weights[iNeuron][iNeuronAbove];
			}
		}
	}*/

	@Override
	public double calculateError(
		double [] realOutput,
		double [] expectedOutput
	) {
		double error = 0.0;
		for (int i = 0 ; i < realOutput.length ; ++i) {
			error += Math.pow(realOutput[i] - expectedOutput[i], 2);
		}
		error *= 0.5;
		return error;
	}
}