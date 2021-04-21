package nnimpl;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class NetworkDataReader {
	public static double[][] loadBasicStuff(String filePath, int numberOfLinesToRead, int numberOfColumns){
		double[][] result = new double[numberOfLinesToRead][numberOfColumns];
		try(BufferedReader br = new BufferedReader(new FileReader(filePath))) {
			String line = null;
			int i = 0;
			while ((line = br.readLine()) != null) {
				String[] stringVector = line.split(",");
				for (int j = 0 ; j < numberOfColumns ; j++) {
					result[i][j] = Double.valueOf(stringVector[j]);
				}
				++i;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	        return result;
	}

	public static double[][] loadMnistInputs(String filePath, int numberOfLinesToRead, int numberOfColumns){
		double[][] result = new double[numberOfLinesToRead][numberOfColumns];
		try(BufferedReader br = new BufferedReader(new FileReader(filePath))) {
			String line = null;
			int i = 0;
			while ((line = br.readLine()) != null) {
				String[] stringVector = line.split(",");
				for (int j = 0 ; j < numberOfColumns ; j++) {
					result[i][j] = Double.valueOf(stringVector[j]) / 255.0;
				}
				++i;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	        return result;
	}

	public static double[][] loadMnistOutputs(String filePath, int numberOfLinesToRead, int numberOfColumns) {
		double[][] result = new double[numberOfLinesToRead][numberOfColumns];

		try(BufferedReader br = new BufferedReader(new FileReader(filePath))) {
			String line = null;
			int i = 0;
			while ((line = br.readLine()) != null) {
				int index = Integer.valueOf(line);
				for (int j = 0 ; j < numberOfColumns ; j++) {
					result[i][j] = 0.0;
				}
				result[i][index] = 1.0;
				++i;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		return result;
	}
}
