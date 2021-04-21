package nnimpl;

import java.util.Random;

public class NormalizedWeightInitialization implements WeightInitialization {
	
	private static Random random = new Random();

	public double [][][] initializeWeights(int [] networkInfo) {
		if (networkInfo == null) {
			return null;
		}
		int nonInputLayers = networkInfo.length - 1;
		double [][][] result = new double[nonInputLayers][][];
		for (int iLayer = 0 ; iLayer < nonInputLayers ; ++iLayer) {
			int inputLayerSize = networkInfo[iLayer];
			int outputLayerSize = networkInfo[iLayer + 1];
			result[iLayer] = new double [inputLayerSize][outputLayerSize];
			for (int iArrowStart = 0 ; iArrowStart < inputLayerSize ; ++iArrowStart) {
				for (int iArrowEnd  = 0 ; iArrowEnd < outputLayerSize ; ++iArrowEnd) {
					// WEIGHT INITIALIZATION
					result[iLayer][iArrowStart][iArrowEnd] = random.nextDouble();
				}
			}
		}
		return result;
	}
}