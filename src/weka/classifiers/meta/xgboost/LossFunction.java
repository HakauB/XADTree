package weka.classifiers.meta.xgboost;

public interface LossFunction {
	
	public double[] getGradients();
	
	public double[] getHessian();
	
	public void setPredictions(double[] predictions);
	
	//public double[] transformPredictions(double[] predictions);
	
	//public double[] performCalculations();
	
}
