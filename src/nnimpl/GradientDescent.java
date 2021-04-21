package nnimpl;
/*
SGD weight config = sum for every example
Momentum = same, but caches previous
*/

public interface GradientDescent {
	public void gradientDescent(double [][][] weightConfiguration, double [][][] epochBuffer);
}