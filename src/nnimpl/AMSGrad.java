package nnimpl;
public class AMSGrad implements GradientDescent {
	
	private double beta1;
	private double beta2;
	private double learningRate;
	private double smoothingTerm;

	private double [][][] mBuffer;
	private double [][][] vBuffer;
	private double [][][] vMaxBuffer;
	
	private double gradientValue;

	public AMSGrad(
		int[] networkInfo, 
		double beta1,
		double beta2,
		double learningRate,
		double smoothingTerm
	) {
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.learningRate = learningRate;
		this.smoothingTerm = smoothingTerm;
		
		this.mBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.vBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);
		this.vMaxBuffer = NetworkBuilder.buildFeedForwardNetworkWeights(networkInfo, false);

		this.gradientValue = 0.0;
	}

	public AMSGrad(
		int[] networkInfo,
		double beta1,
		double beta2,
		double learningRate
	) {
		this(networkInfo, beta1, beta2, learningRate,  -8);
	}	

	public AMSGrad(
		int[] networkInfo,
		double beta1,
		double beta2
	) {
		this(networkInfo, beta1, beta2,  0.01);
	}

	public AMSGrad(
		int[] networkInfo,
		double beta1
	) {
		this(networkInfo, beta1,  0.999);
	}

	public AMSGrad(
		int[] networkInfo
	) {
		this(networkInfo,  0.9);
	}

	@Override
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer) {
		for (int iLayer = 0 ; iLayer < weightConfiguration.length ; ++iLayer) {
			for (int iInputNeuron = 0 ; iInputNeuron < weightConfiguration[iLayer].length ; ++iInputNeuron) {
				for (int iOuputNeuron = 0 ; iOuputNeuron < weightConfiguration[iLayer][iInputNeuron].length ; ++iOuputNeuron) {
					gradientValue = epochBuffer[iLayer][iInputNeuron][iOuputNeuron];
					
					mBuffer[iLayer][iInputNeuron][iOuputNeuron] *= beta1;
					mBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - beta1) * gradientValue;

					vBuffer[iLayer][iInputNeuron][iOuputNeuron] *= beta2;
					vBuffer[iLayer][iInputNeuron][iOuputNeuron] += (1 - beta2) * gradientValue * gradientValue;
					
					vMaxBuffer[iLayer][iInputNeuron][iOuputNeuron] = Math.max(vMaxBuffer[iLayer][iInputNeuron][iOuputNeuron], vBuffer[iLayer][iInputNeuron][iOuputNeuron]);

					weightConfiguration[iLayer][iInputNeuron][iOuputNeuron] -= 
						(learningRate * mBuffer[iLayer][iInputNeuron][iOuputNeuron]) 
						/ Math.sqrt(vMaxBuffer[iLayer][iInputNeuron][iOuputNeuron] + smoothingTerm);
				}
			}
		}
	}
}
