package nnimpl;
import java.util.Random;
import java.lang.Math;

public class HeEtAlWeightInitialization implements WeightInitialization {
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
					//w= np.random.randn( layer_size[l],layer_size[l-1] ) * np.sqrt( 2/ layer_size[l-1])
					result[iLayer][iArrowStart][iArrowEnd] = random.nextDouble() * Math.sqrt(2 / networkInfo[iLayer]);
				}
			}
		}
		return result;
	}
}