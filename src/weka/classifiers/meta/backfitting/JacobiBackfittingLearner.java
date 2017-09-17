package weka.classifiers.meta.backfitting;

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

public class JacobiBackfittingLearner extends AbstractClassifier {

	private int m_NumTrees = 10;
	private int m_NumIterations = 50;

	private Classifier[] m_Classifiers;
	private Classifier m_Classifier;

	private Instances m_Data;

	public JacobiBackfittingLearner() {
		m_Classifier = new DecisionStump();
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		m_Data = data;
		m_Classifiers = AbstractClassifier.makeCopies(m_Classifier, m_NumTrees);

		double RMSETotal = 0.0;
		double[] treeRMSEs = new double[m_NumTrees];

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

		// For number of user specified iterations
		for (int passage = 0; passage < m_NumIterations; passage++) {
			Classifier[] newClassifiers = new Classifier[m_Classifiers.length];

			for (int tree = 0; tree < m_NumTrees; tree++) {
				Instances newTrainSet = new Instances(data);
				
				/*double[] predSum = new double[data.numInstances()];
				for (int i = 0; i < m_NumTrees; i++) {
					if (tree != i) {
						for (int j = 0; j < data.numInstances(); j++) {
							predSum[j] += m_Classifiers[i].classifyInstance(data.instance(j));
						}
					}
				}*/
				
				double[] currPreds = new double[data.numInstances()];
				for(int i = 0; i < data.numInstances(); i++)
				{
					currPreds[i] = m_Classifiers[tree].classifyInstance(data.instance(i));
				}

				for (int i = 0; i < newTrainSet.numInstances(); i++) {
					newTrainSet.instance(i).setClassValue(data.instance(i).classValue() - (predictionSum[i] - currPreds[i]));
				}
				// if (pass == 0) {
				newClassifiers[tree] = new DecisionStump();
				newClassifiers[tree].buildClassifier(newTrainSet);
				currPreds = null;
			}
			
			predictionSum = new double[data.numInstances()];
			
			for (int tree = 0; tree < m_NumTrees; tree++) {
				
				for(int i = 0; i < data.numInstances(); i++) {
					predictionSum[i] += newClassifiers[tree].classifyInstance(data.instance(i));
				}
				
			}
			//m_Classifiers = newClassifiers;
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
		runClassifier(new JacobiBackfittingLearner(), args);
	}
}
