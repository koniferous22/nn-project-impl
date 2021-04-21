package nnimpl;
public class CrossEntropy implements ErrorFunction {
    @Override
    public void derivationOutputLayer(
        double [] realOutput, 
        double [] expectedOutput,
        double [] derivationOutput
    ) {
        for (int i = 0 ; i < realOutput.length ; ++i) {
            derivationOutput[i] = -expectedOutput[i] / (realOutput[i] + 0.001) + (1 - expectedOutput[i]) / (1.001 - realOutput[i]);
        }
    }

    @Override
    public double calculateError(
        double [] realOutput,
        double [] expectedOutput
    ) {
        double error = 0.0;
        for (int i = 0 ; i < realOutput.length ; ++i) {
            error += expectedOutput[i] * Math.log(realOutput[i])
                    + (1 - expectedOutput[i]) * Math.log(1 - realOutput[i]);
        }
        return -error;
    }
}
