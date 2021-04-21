// JE KURVA SILNO MOZNE ZE PRVA EPOCHA BUDE KOPMLET NA KEKET
package nnimpl;

public class AdaDelta implements GradientDescent {
	
	private double lambdaRMS;
	private double lambdaAdaDelta;
	private double learningRate;
	private double smoothingTerm;

	private double [][][] rmsPropBuffer;
	private double [][][] adaDeltaBuffer;
	
	private double gradientValue;
	private double rmsGradMenovathel;
	private double parameterUpdate;

	public AdaDelta(
		int[] networkInfo,
		double lambdaRMS,
		double lambdaAdaDelta,
		double learningRate,
		double smoothingTerm
	) {
		this.rmsPropBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.adaDeltaBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);

		this.lambdaRMS = lambdaRMS;
		this.lambdaAdaDelta = lambdaAdaDelta;
		this.learningRate = learningRate;
		this.smoothingTerm = smoothingTerm;

		this.gradientValue = 0.0;
	}

	public AdaDelta(int[] networkInfo) {
		this(
			networkInfo,
			0.9,
			0.9,
			0.01,
			Math.exp(-0.8)
		);
	}


	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {		

		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					//System.out.printf("epochBuffer[%d][%d][%d] = %f", iLayer, iInputNeuron, iOuputNeuron, epochBuffer[iLayer][iInputNeuron][iOuputNeuron]);

					gradientValue = epochBuffer[iLayer][iInputNeuron][iOuputNeuron];
					
					rmsPropBuffer[iLayer][iInputNeuron][iOuputNeuron] *= lambdaRMS;
					rmsPropBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - lambdaRMS) * gradientValue * gradientValue;
					
					rmsGradMenovathel = Math.sqrt(rmsPropBuffer[iLayer][iInputNeuron][iOuputNeuron] + smoothingTerm);
					parameterUpdate = (learningRate * gradientValue) / rmsGradMenovathel;
					
					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= 
						(Math.sqrt(adaDeltaBuffer[iLayer][iInputNeuron][iOuputNeuron] + smoothingTerm) * gradientValue) / 
						rmsGradMenovathel;

					// UPDATE
					adaDeltaBuffer[iLayer][iInputNeuron][iOuputNeuron] *= lambdaAdaDelta;
					adaDeltaBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - lambdaAdaDelta) * parameterUpdate * parameterUpdate;

					//System.out.println("===");
				}
			}
		}

	}
}
