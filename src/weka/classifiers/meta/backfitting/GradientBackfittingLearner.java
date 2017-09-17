package weka.classifiers.meta.backfitting;

import java.util.Arrays;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.meta.xgboost.XGBTree;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

public class GradientBackfittingLearner extends AbstractClassifier {

	private int m_NumTrees = 50;
	private int m_NumIterations = 50;

	private Attribute m_ClassAttribute;
	private XGBTree[] m_Classifiers;

	private Instances m_Data;

	private double m_LearningRate = 0.3;

	Attribute m_PredictionAttribute;
	Attribute m_GradientAttribute;
	Attribute m_HessianAttribute;
	double[] predicts;

	double m_lambda;
	double m_lambdaBias;
	double m_alpha;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(data);
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		m_NumTrees = 50;
		m_Classifiers = new XGBTree[m_NumTrees];
		for (int c = 0; c < m_NumTrees; c++) {
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

		for (int a = 0; a < m_NumTrees; a++) {
			XGBTree temp = new XGBTree();
			temp.setLearningRate(m_LearningRate);
			m_Classifiers[a] = temp;
			m_Classifiers[a].buildClassifier(data);
			// if(a > 0) {
			for (int j = 0; j < data.numInstances(); j++) {
				double oldPred = data.instance(j).value(m_PredictionAttribute);
				double newWeight = m_Classifiers[a].weightForInstance(data.instance(j));
				data.instance(j).setValue(m_PredictionAttribute, oldPred + newWeight);
			}
			m_Data = data;
		}

		// Get initial prediction total
		double[] predictionSum = new double[data.numInstances()];
		for (int i = 0; i < m_NumTrees; i++) {
			for (int j = 0; j < data.numInstances(); j++) {
				predictionSum[j] += m_Classifiers[i].weightForInstance(data.instance(j));
			}
		}

		int curTree = 0;
		int iter = 0;

		while (iter < m_NumIterations) {
			Instances newTrainSet = new Instances(data);

			double[] currPred = new double[data.numInstances()];
			for (int i = 0; i < data.numInstances(); i++) {
				currPred[i] = m_Classifiers[curTree].weightForInstance(data.instance(i));
			}

			for (int i = 0; i < newTrainSet.numInstances(); i++) {
				newTrainSet.instance(i).setClassValue(data.instance(i).classValue() - (predictionSum[i] - currPred[i]));
			}
			m_Classifiers[curTree] = new XGBTree();
			m_Classifiers[curTree].buildClassifier(newTrainSet);

			for (int i = 0; i < data.numInstances(); i++) {
				predictionSum[i] -= currPred[i];
				predictionSum[i] += m_Classifiers[curTree].weightForInstance(data.instance(i));
			}

			iter++;
			curTree++;
			if (curTree == m_NumTrees) {
				curTree = 0;
			}
			currPred = null;
			newTrainSet = null;
		}
	}

	@Override
	public String toString() {
		StringBuffer text = new StringBuffer();
		for (int i = 0; i < m_Classifiers.length; i++) {
			text.append("Tree: " + i + " = " + m_Classifiers[i].toString() + "\n\n");
		}
		return text.toString();
	}

	@Override
	public double classifyInstance(Instance inst) throws Exception {
		double score = 0.0;
		for (int i = 0; i < m_Classifiers.length - 1; i++) {
			score += m_Classifiers[i].weightForInstance(inst);
		}
		return score;
	}

	private double[] buildAlphas(int max) {
		double[] alphaArray = new double[max];
		double pass = 1;
		for (int i = 0; i < alphaArray.length; i++) {
			if (i % 3 == 0) {
				alphaArray[i] = 5 / (pass * 10);
			} else if (i % 3 == 1) {
				alphaArray[i] = 2 / (pass * 10);
			} else if (i % 3 == 2) {
				alphaArray[i] = 1 / (pass * 10);
				pass *= 10;
			}
		}
		return alphaArray;
	}

	public static void main(String[] args) {
		runClassifier(new GradientBackfittingLearner(), args);
	}
}
