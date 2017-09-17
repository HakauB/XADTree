package weka.classifiers.meta;

import weka.classifiers.IteratedSingleClassifierEnhancer;
import weka.classifiers.meta.xgboost.XGBTree;
import weka.core.*;
import weka.core.Capabilities.Capability;

public class XgbLearner extends IteratedSingleClassifierEnhancer {

	private Instances m_Data;
	
	/** The node's successors. */
	private XgbLearner[] m_Successors;

	private XGBTree[] m_Classifiers;

	private XGBTree m_Classifier;

	private int depth = 0;

	private int iterations = 1;

	private int m_Seed = 1;

	/** Attribute used for splitting. */
	private Attribute m_Attribute;

	/** Class value if node is leaf. */
	private double m_ClassValue;

	/** Class distribution if node is leaf. */
	private double[] m_Distribution;

	/** Class attribute of dataset. */
	private Attribute m_ClassAttribute;

	private double m_splitValue;
	
	private double m_LearningRate = 1.0;
	
	private boolean m_useBackfitting = true;

	
	Attribute m_PredictionAttribute;
	Attribute m_GradientAttribute;
	Attribute m_HessianAttribute;
	double[] predicts;

	double m_lambda = 1.0;
	double m_lambdaBias;
	double m_alpha = 0.0;
	
	@OptionMetadata(displayName = "Learning Rate", description = "Controls learning rate of boost", commandLineParamName = "learningRate", commandLineParamSynopsis = "-learningRate", displayOrder = 2)
	public void setLearningRate(double newRate) {
		m_LearningRate = newRate;
	}

	public double getLearningRate() {
		return m_LearningRate;
	}
	
	@OptionMetadata(displayName = "lambda", description = "Controls L2 regularization", commandLineParamName = "lambda", commandLineParamSynopsis = "-lambda", displayOrder = 3)
	public void setLambda(double lambda) {
		m_lambda = lambda;
	}

	public double getLambda() {
		return m_lambda;
	}
	
	@OptionMetadata(displayName = "boostIterations", description = "Number of boosting rounds", commandLineParamName = "boostIterations", commandLineParamSynopsis = "-boostIterations", displayOrder = 4)
	public void setIterations(int iter) {
		iterations = iter;
	}

	public double getIterations() {
		return iterations;
	}

	public XgbLearner() {
		m_Classifier = new XGBTree();
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		iterations = 100;
		m_Classifiers = new XGBTree[iterations];
		for(int c = 0; c < iterations; c++) {
			m_Classifiers[c] = new XGBTree();
		}
		data.insertAttributeAt(new Attribute("prediction"), data.numAttributes());
		data.insertAttributeAt(new Attribute("gradient"), data.numAttributes());
		data.insertAttributeAt(new Attribute("hessian"), data.numAttributes());
		m_ClassAttribute = data.attribute(data.numAttributes() - 4);
		m_PredictionAttribute = data.attribute(data.numAttributes() - 3);
		m_GradientAttribute = data.attribute(data.numAttributes() - 2);
		m_HessianAttribute = data.attribute(data.numAttributes() - 1);
		for (int i = 0; i < data.numInstances(); i++) {
			data.instance(i).setValue(m_PredictionAttribute, 0.0);
			data.instance(i).setValue(m_GradientAttribute, 0.0);
			data.instance(i).setValue(m_HessianAttribute, 0.0);
		}
		
		for(int a = 0; a < iterations; a++) {
			XGBTree temp = new XGBTree();
			temp.setLearningRate(m_LearningRate);
			temp.setLambda(m_lambda);
			m_Classifiers[a] = temp;
			m_Classifiers[a].buildClassifier(data);
			//if(a > 0) {
			for(int j = 0; j < data.numInstances(); j++) {
				double oldPred = data.instance(j).value(m_PredictionAttribute);
				double newWeight = m_Classifiers[a].weightForInstance(data.instance(j));
				data.instance(j).setValue(m_PredictionAttribute, oldPred + newWeight);
			}
			m_Data = data;
		}
		
		if(m_useBackfitting) {
			int m = m_Classifiers.length;
			for(int b = 0; b < iterations; b++) {
				
				for(int j = 0; j < data.numInstances(); j++) {
					double oldPred = data.instance(j).value(m_PredictionAttribute);
					double newWeight = m_Classifiers[b%m].weightForInstance(data.instance(j));
					data.instance(j).setValue(m_PredictionAttribute, oldPred - newWeight);
				}
				
				m_Classifiers[b%m].buildClassifier(data);
				
				for(int j = 0; j < data.numInstances(); j++) {
					double oldPred = data.instance(j).value(m_PredictionAttribute);
					double newWeight = m_Classifiers[b%m].weightForInstance(data.instance(j));
					data.instance(j).setValue(m_PredictionAttribute, oldPred + newWeight);
				}
			}
		}
	}

	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NUMERIC_ATTRIBUTES);

		// class
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	@Override
	public double classifyInstance(Instance inst) {
		double sum = 0.0;
		for(int i = 0; i < iterations; i++) {
			sum += m_Classifiers[i].weightForInstance(inst);
		}
		if(m_ClassAttribute.isNominal()){
			if(sum < 0.5) {
				return 0;
			} else {
				return 1;
			}
		}
		return sum;
	}
	
	@Override
	public String toString() {
		StringBuffer text = new StringBuffer();
		for(int i = 0; i < iterations; i++) {
			text.append("Tree: " + i + " = " + m_Classifiers[i].toString() + "\nWeight sum: " + m_Classifiers[i].weightSum() + "\n\n");
		}
		return text.toString();
	}

	private double calcDelta(double sum_grad, double sum_hess, double w) {
		if (sum_hess < 1E-5) {
			return 0.0;
		}
		double temp = w - (sum_grad + m_lambda * w) / (sum_hess + m_lambda);

		if (temp >= 0) {
			return Math.max(-(sum_grad + m_lambda * w + m_alpha) / (sum_hess + m_lambda), -w);
		} else {
			return Math.min(-(sum_grad + m_lambda * w - m_alpha) / (sum_hess + m_lambda), -w);
		}
	}

	private double calcDeltaBias(double sum_grad, double sum_hess, double w) {
		return -(sum_grad + m_lambdaBias * w) / (sum_hess + m_lambdaBias);
	}

	private double sum(double[] array) {
		double total = 0.0;
		for (int i = 0; i < array.length; i++) {
			total += array[i];
		}
		return total;
	}

	public static void main(String[] args) {
		runClassifier(new XgbLearner(), args);
	}
}
