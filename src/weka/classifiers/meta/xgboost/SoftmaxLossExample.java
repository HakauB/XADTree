package weka.classifiers.meta.xgboost;

import java.util.ArrayList;

import weka.core.Instances;

public class SoftmaxLossExample implements LossFunction {

	private Instances data;
	private double[] preds;
	String fName;
	int numClasses;
	
	public SoftmaxLossExample(Instances data, double[] predictions, int numClasses, String name) {
		this.data = data;
		preds = predictions;
		fName = name;
		numClasses = data.numClasses();
	}
	
	public void performCalculation() {
		int nData = data.numInstances();
		ArrayList<Double> rec = new ArrayList<>();
		for(int i = 0; i < nData; i++) {
			for(int k = 0; k < numClasses; k++) {
				rec.add(preds[i * numClasses + k]);
			}
		}
		
	}
	
	@Override
	public double[] getGradients() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] getHessian() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setPredictions(double[] predictions) {
		// TODO Auto-generated method stub
		
	}

}
