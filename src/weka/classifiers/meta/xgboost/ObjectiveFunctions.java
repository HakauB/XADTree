package weka.classifiers.meta.xgboost;

import weka.core.Instances;
import weka.core.Attribute;

public class ObjectiveFunctions implements LossFunction {

	private boolean initialized = false;
	private String baseFunction;

	Attribute predictionAtt;
	Attribute gradientAtt;
	Attribute hessianAtt;

	double[] labels;
	double[] preds;
	double[] transformedPreds;
	double[] gradients;
	double[] hessian;

	double error;

	public ObjectiveFunctions(Instances data, double[] predicts, String func) {
		baseFunction = func;
		preds = predicts;
		runCalculations(data);
	}

	public ObjectiveFunctions(Instances data, Attribute predictionAttribute, String func) {
		baseFunction = func;
		preds = null;
		predictionAtt = predictionAttribute;
		runCalculations(data);
	}
	
	public ObjectiveFunctions(Instances data, Attribute predictionAttribute, Attribute gradAtt, Attribute hessAtt, String func) {
		baseFunction = func;
		preds = null;
		predictionAtt = predictionAttribute;
		gradientAtt = gradAtt;
		hessianAtt = hessAtt;
		runCalculations(data);
	}

	public void runCalculations(Instances data) {
		if (!initialized) {
			transformedPreds = new double[data.numInstances()];

			if (baseFunction.equals("rank"))
				buildRanking(data);
			else if (baseFunction.equals("logloss"))
				buildLogLoss(data);
			else if (baseFunction.equals("linloss"))
				buildLinearLoss(data);
		}
	}

	public void buildLinearLoss(Instances data) {
		if (preds == null) {
			for (int i = 0; i < data.numInstances(); i++) {
				data.instance(i).setValue(gradientAtt, data.instance(i).value(predictionAtt) - data.instance(i).classValue());
				data.instance(i).setValue(hessianAtt, 1.0);
			}
		} else {
			gradients = new double[preds.length];
			hessian = new double[preds.length];
			for (int i = 0; i < gradients.length; i++) {
				gradients[i] = preds[i] - labels[i];
				hessian[i] = 1.0;
			}
		}
	}

	public void buildRanking(Instances data) {
		Instances dataCopy = new Instances(data);
		data.sort(data.classAttribute());

		double[] lambdas = new double[data.numInstances()];
		double[] delts = new double[data.numInstances()];
		double[][] lambdaMatrix = new double[data.numInstances()][data.numInstances()];

		// May implement dcg for lambdamart at some point
		double[][] dcgMatrix = new double[data.numInstances()][data.numInstances()];

		double scale = 1.0 / data.numInstances(); // Pairs generated for each
													// instance

		double weight = 1.0 * scale;

		for (int i = 0; i < data.numInstances(); i++) {

			for (int j = 0; j < data.numInstances(); j++) {
				double rho = lambdaMatrix[i][j] = weight
						/ (1 + Math.exp(data.instance(i).value(predictionAtt)) - data.instance(j).value(predictionAtt));
				lambdas[i] += rho - 1;
				lambdas[j] -= rho - 1;

				delts[i] += 2 * rho * (1.0 - rho);
				delts[j] += 2 * rho * (1.0 - rho);
			}
		}
		gradients = lambdas;
		hessian = delts;
	}

	// Cheaper to perform an update than rebuild in some scenarios
	public void updateRanking(Instances data, double[] predictions) {

	}

	public void buildLogLoss(Instances data) {
		// Transform prediction values
		/*for (int i = 0; i < data.numInstances(); i++) {
			transformedPreds[i] = 1.0 / (1.0 + Math.exp(-data.instance(i).value(predictionAtt)));
		}

		// Calculate gradients
		gradients = new double[transformedPreds.length];
		for (int i = 0; i < gradients.length; i++) {
			gradients[i] = transformedPreds[i] - data.instance(i).classValue();
		}

		// Calculate hessian
		hessian = new double[transformedPreds.length];
		for (int i = 0; i < hessian.length; i++) {
			hessian[i] = transformedPreds[i] * (1 - transformedPreds[i]);
		}*/
		
		if (preds == null) {
			for (int i = 0; i < data.numInstances(); i++) {
				double transformedPred = 1.0 / (1.0 + Math.exp(-data.instance(i).value(predictionAtt)));
				data.instance(i).setValue(gradientAtt, transformedPred - data.instance(i).classValue());
				data.instance(i).setValue(hessianAtt, transformedPred * (1 - transformedPred));
			}
		} else {
			gradients = new double[preds.length];
			hessian = new double[preds.length];
			for (int i = 0; i < gradients.length; i++) {
				gradients[i] = preds[i] - labels[i];
				hessian[i] = 1.0;
			}
		}
	}
	
	public void buildSoftMaxLoss(Instances data) {
		
	}

	public double[] getGradients() {
		return gradients;
	}

	public double[] getHessian() {
		return hessian;
	}

	public double getError() {
		return error;
	}

	public double[] getLabels(Instances data) {
		double[] labels = new double[data.numInstances()];
		for (int i = 0; i < data.numInstances(); i++) {
			labels[i] = data.instance(i).classValue();
		}
		return labels;
	}

	@Override
	public void setPredictions(double[] predictions) {
		preds = predictions;

	}
}
