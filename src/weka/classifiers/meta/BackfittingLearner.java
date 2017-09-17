package weka.classifiers.meta;

import java.util.Arrays;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionMetadata;

public class BackfittingLearner extends AbstractClassifier {

	private int m_NumTrees = 30;
	private int m_NumIterations = 5000;

	private Classifier[] m_Classifiers;
	private Classifier m_Classifier;

	private Instances m_Data;
	
	@OptionMetadata(displayName = "Trees", description = "Number of trees", commandLineParamName = "nTrees", commandLineParamSynopsis = "-nTrees", displayOrder = 2)
	public void setNumTrees(int nTrees) {
		m_NumTrees = nTrees;
	}

	public int getNumTrees() {
		return m_NumTrees;
	}
	
	@OptionMetadata(displayName = "Iterations", description = "Number of iterations", commandLineParamName = "iterations", commandLineParamSynopsis = "-iterations", displayOrder = 3)
	public void setNumIterations(int iters) {
		m_NumTrees = iters;
	}

	public int getNumIterations() {
		return m_NumIterations;
	}

	public BackfittingLearner() {
		m_Classifier = new DecisionStump();
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		m_Data = data;
		m_Classifiers = AbstractClassifier.makeCopies(m_Classifier, m_NumTrees);

		// Train initial tree set
		Instances initialData = new Instances(data);
		for (int i = 0; i < m_NumTrees; i++) {
			m_Classifiers[i].buildClassifier(initialData);
			for (int j = 0; j < initialData.numInstances(); j++) {
				double oldValue = initialData.instance(j).classValue();
				double newValue = m_Classifiers[i].classifyInstance(initialData.instance(j));
				initialData.instance(j).setClassValue(oldValue - newValue);
			}
		}

		// Get initial prediction total
		double[] predictionSum = new double[data.numInstances()];
		for (int i = 0; i < m_NumTrees; i++) {
			for (int j = 0; j < data.numInstances(); j++) {
				predictionSum[j] += m_Classifiers[i].classifyInstance(data.instance(j));
			}
		}

		int curTree = 0;
		int iter = 0;

		while (iter < m_NumIterations) {
			Instances newTrainSet = new Instances(data);

			double[] currPred = new double[data.numInstances()];
			for(int i = 0; i < data.numInstances(); i++) {
				currPred[i] = m_Classifiers[curTree].classifyInstance(data.instance(i));
			}
			

			for (int i = 0; i < newTrainSet.numInstances(); i++) {
				newTrainSet.instance(i).setClassValue(data.instance(i).classValue() - (predictionSum[i] - currPred[i]));
			}
			m_Classifiers[curTree] = new DecisionStump();
			m_Classifiers[curTree].buildClassifier(newTrainSet);
			
			for(int i = 0; i < data.numInstances(); i++) {
				predictionSum[i] -= currPred[i];
				predictionSum[i] += m_Classifiers[curTree].classifyInstance(data.instance(i));
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
		for (int i = 0; i < m_NumTrees; i++) {
			text.append("Tree: " + i + " = " + m_Classifiers[i].toString() + "\n\n");
		}
		return text.toString();
	}

	@Override
	public double classifyInstance(Instance inst) throws Exception {
		double score = 0.0;
		for (int i = 0; i < m_Classifiers.length - 1; i++) {
			score += m_Classifiers[i].classifyInstance(inst);
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
		runClassifier(new BackfittingLearner(), args);
	}
}
