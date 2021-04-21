package nnimpl;

import java.util.Random;
import java.lang.Math;

public class NetworkBuilder {
	private static Random random = new Random();

	public static double[][] buildFeedForwardNetworkNeurons(int[] networkInfo) {
		return buildFeedForwardNetworkNeurons(networkInfo, true);
	}

	public static double[][] buildFeedForwardNetworkNeurons(int[] networkInfo, boolean includeFormalInput) {
		if (networkInfo == null) {
			return null;
		}

		double [][] result = new double[networkInfo.length][];

		if (includeFormalInput) {
			for (int iLayer = 0 ; iLayer < networkInfo.length ; ++iLayer) {
				result[iLayer] = new double[networkInfo[iLayer]];
				for (int iNeuron = 0 ; iNeuron < networkInfo[iLayer] ; ++iNeuron) {
					result[iLayer][iNeuron] = 0.0;
				}
			}	
		} else {
			for (int iLayer = 1 ; iLayer < networkInfo.length ; ++iLayer) {
				result[iLayer - 1] = new double[networkInfo[iLayer]];
				for (int iNeuron = 0 ; iNeuron < networkInfo[iLayer] ; ++iNeuron) {
					result[iLayer - 1][iNeuron] = 0.0;
				}
			}
		}

		return result;
	}

	public static double[][][] buildFeedForwardNetworkWeights(int [] networkInfo) {
		return buildFeedForwardNetworkWeights(networkInfo, true);
	}

	public static double[][][] buildFeedForwardNetworkWeights(int [] networkInfo, boolean randomizeContents) {
		if (networkInfo == null) {
			return null;
		}
		int nonInputLayers = networkInfo.length - 1;
		double [][][] result = new double[nonInputLayers][][];
		if (randomizeContents) {
			for (int iLayer = 0 ; iLayer < nonInputLayers ; ++iLayer) {
				int inputLayerSize = networkInfo[iLayer];
				int outputLayerSize = networkInfo[iLayer + 1];
				result[iLayer] = new double [inputLayerSize][outputLayerSize];
				for (int iArrowStart = 0 ; iArrowStart < inputLayerSize ; ++iArrowStart) {
					for (int iArrowEnd  = 0 ; iArrowEnd < outputLayerSize ; ++iArrowEnd) {
						// WEIGHT INITIALIZATION
						result[iLayer][iArrowStart][iArrowEnd] = random.nextGaussian() * Math.sqrt(6.0 / (inputLayerSize + outputLayerSize));
					}
				}
			}
		} else {
			for (int iLayer = 0 ; iLayer < nonInputLayers ; ++iLayer) {
				int inputLayerSize = networkInfo[iLayer];
				int outputLayerSize = networkInfo[iLayer + 1];
				result[iLayer] = new double [inputLayerSize][outputLayerSize];
				for (int iArrowStart = 0 ; iArrowStart < inputLayerSize ; ++iArrowStart) {
					for (int iArrowEnd  = 0 ; iArrowEnd < outputLayerSize ; ++iArrowEnd) {
						// WEIGHT INITIALIZATION
						result[iLayer][iArrowStart][iArrowEnd] = 0.0;
					}
				}
			}
		}
		return result;
	}

	public static int[][] buildFeedForwardDropouts(int [] networkInfo) {
		if (networkInfo == null) {
			return null;
		}
		int [][] result = new int[networkInfo.length - 1][];
		for (int iLayer = 1 ; iLayer < networkInfo.length ; ++iLayer) {
			result[iLayer - 1] = new int[networkInfo[iLayer]];
			for (int iNeuron = 0 ; iNeuron < networkInfo[iLayer] ; ++iNeuron) {
				result[iLayer - 1][iNeuron] = 0;
			}
		}
		return result;
	}
}