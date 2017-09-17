package weka.classifiers.meta.backfitting;

import java.util.Enumeration;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.xgboost.ObjectiveFunctions;
import weka.core.*;
import weka.core.Capabilities.Capability;

public class BackfitTree extends AbstractClassifier {

	private static final long serialVersionUID = -6137432841812541005L;

	/** The node's successors. */
	private BackfitTree[] m_Successors;

	/** Attribute used for splitting. */
	private Attribute m_Attribute;

	/** Class value if node is leaf. */
	private double m_ClassValue;

	/** Class distribution if node is leaf. */
	private double[] m_Distribution;

	/** Class attribute of dataset. */
	private Attribute m_ClassAttribute;

	private double m_splitValue;

	Attribute m_PredictionAttribute;
	Attribute m_GradientAttribute;
	Attribute m_HessianAttribute;
	
	int totalDataSize;
	
	double alpha = 0.5;
	
	boolean leaf;
	
	@OptionMetadata(displayName = "Alpha", description = "Controls tree size", commandLineParamName = "alpha", commandLineParamSynopsis = "-alpha", displayOrder = 3)
	public void setAlpha(double newAlpha) {
		alpha = newAlpha;
	}

	public double getLambda() {
		return alpha;
	}

	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		if (data.classAttribute().equals(data.attribute(data.numAttributes() - 1))) {
			// System.out.println("Class is last");
			data.insertAttributeAt(new Attribute("prediction"), data.numAttributes());
			data.insertAttributeAt(new Attribute("gradient"), data.numAttributes());
			data.insertAttributeAt(new Attribute("hessian"), data.numAttributes());
			m_PredictionAttribute = data.attribute(data.numAttributes() - 3);
			m_GradientAttribute = data.attribute(data.numAttributes() - 2);
			m_HessianAttribute = data.attribute(data.numAttributes() - 1);
			for (int i = 0; i < data.numInstances(); i++) {
				data.instance(i).setValue(m_PredictionAttribute, 0.0);
				data.instance(i).setValue(m_GradientAttribute, 0.0);
				data.instance(i).setValue(m_HessianAttribute, 0.0);
			}
		} else {
			m_PredictionAttribute = data.attribute(data.numAttributes() - 3);
			m_GradientAttribute = data.attribute(data.numAttributes() - 2);
			m_HessianAttribute = data.attribute(data.numAttributes() - 1);
		}
		
		totalDataSize = data.numInstances();

		makeGreedyTree(data);
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

	public void makeGreedyTree(Instances data) throws Exception {
		
		m_ClassAttribute = data.classAttribute();

		// Check if no instances have reached this node.
		if (data.numInstances() == 0) {
			m_Attribute = null;
			m_splitValue = Double.MAX_VALUE;
			m_ClassValue = Utils.missingValue();
			m_Distribution = new double[data.numClasses()];
			System.out.println("BAD");
			return;
		}

		ObjectiveFunctions f = new ObjectiveFunctions(data, m_PredictionAttribute, m_GradientAttribute,
				m_HessianAttribute, "linloss");

		double gain = 0.0;
		double gradientSum = data.attributeStats(data.numAttributes() - 2).numericStats.sum;
		double hessianSum = data.attributeStats(data.numAttributes() - 1).numericStats.sum;

		for (int a = 0; a < data.numAttributes() - 4; a++) {
			double gradLeftSum = 0;
			double gradRightSum = gradientSum;
			double hessLeftSum = 0;
			double hessRightSum = hessianSum;

			data.sort(a);

			for (int i = 0; i < data.numInstances() - 1; i++) {
				gradLeftSum += data.instance(i).value(m_GradientAttribute);
				gradRightSum = gradientSum - gradLeftSum;

				hessLeftSum += data.instance(i).value(m_HessianAttribute);
				hessRightSum = hessianSum - hessLeftSum;

				double newScore = (Math.pow(gradLeftSum, 2.0) / hessLeftSum) + (Math.pow(gradRightSum, 2.0) / hessRightSum)
										- (Math.pow(gradientSum, 2.0) / hessianSum);
				if (data.instance(i).value(a) != data.instance(i + 1).value(a)) {
					if (newScore > gain) {
						gain = newScore;
						m_Attribute = data.attribute(a);
						m_splitValue = (data.instance(i).value(a) + data.instance(i + 1).value(a)) / 2;
					}
				}
			}
		}
		// System.out.println(m_Attribute + " " + m_splitValue + " " + gain);

		//weight = -gradientSum / 2 * hessianSum;

		if ((double)data.numInstances() / (double)totalDataSize <= alpha || Utils.eq(gain, 0.0) || m_Attribute == null) {
			leaf = true;
			m_Attribute = null;
			
			if (m_ClassAttribute.isNumeric()) {
				m_Distribution = new double[1];
				m_ClassValue = data.attributeStats(data.classIndex()).numericStats.mean;
				m_Distribution[0] = m_ClassValue;
			} else if (m_ClassAttribute.isNominal()) {
				m_Distribution = new double[data.numClasses()];
				Enumeration<Instance> instEnum = data.enumerateInstances();
				while (instEnum.hasMoreElements()) {
					Instance inst = instEnum.nextElement();
					m_Distribution[(int) inst.classValue()]++;
				}
				Utils.normalize(m_Distribution);
				m_ClassValue = Utils.maxIndex(m_Distribution);
				m_splitValue = Double.MAX_VALUE;
			}
			
		} else {
			m_Successors = new BackfitTree[2];
			Instances[] split = splitData(data, m_Attribute, m_splitValue);
			for (int c = 0; c < 2; c++) {
				m_Successors[c] = new BackfitTree();
				m_Successors[c].m_PredictionAttribute = m_PredictionAttribute;
				m_Successors[c].m_GradientAttribute = m_GradientAttribute;
				m_Successors[c].m_HessianAttribute = m_HessianAttribute;
				m_Successors[c].totalDataSize = totalDataSize;
				m_Successors[c].makeGreedyTree(split[c]);
			}
		}
		// System.out.println("Done");
	}

	private Instances[] splitData(Instances data, Attribute att, double splitValue) {
		Instances[] split = new Instances[2];
		split[0] = new Instances(data, data.numInstances());
		split[1] = new Instances(data, data.numInstances());

		// Check each instance
		for (int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.instance(i);
			// Less than split, go to left
			if (inst.value(att) < splitValue) {
				split[0].add(inst);
			} else { // Otherwise, go to right
				split[1].add(inst);
			}
		}
		return split;
	}
	
	/*private double L1Threshold(double weight, double lambda) {
		if(weight > lambda) {
			return weight - lambda;
		} else if (weight < -lambda) {
			return weight + lambda;
		}
		return 0.0;
	}
	
	private double calcWeight(double sumGrad, double sumHess) {
		if(sumHess < m_MinChildWeight) {
			return 0.0;
		}
		double dw = 0.0;
		if(m_RegAlpha == 0.0) {
			dw = -sumGrad / (sumHess + m_RegLambda);
		} else {
			dw = -L1Threshold(sumGrad, m_RegAlpha) / (sumHess + m_RegLambda);
		}
		if(m_MaxDeltaStep != 0.0) {
			if(dw > m_MaxDeltaStep) {
				dw = m_MaxDeltaStep;
			} if(dw < -m_MaxDeltaStep) {
				dw = -m_MaxDeltaStep;
			}
		}
		return dw;
	}
	
	private double calcGain(double sumGrad, double sumHess) {
		if(sumHess < m_MinChildWeight) {
			return 0.0;
		}
		if(m_MaxDeltaStep == 0.0) {
			if(m_RegAlpha == 0.0) {
				return sqr(sumGrad) / (sumHess + m_RegLambda);
			} else {
				return sqr(L1Threshold(sumGrad, m_RegAlpha)) / (sumHess + m_RegLambda);
			}
		} else {
			double weight = calcWeight(sumGrad, sumHess);
			double ret = sumGrad * weight + 0.5 * (sumHess + m_RegLambda) * sqr(weight);
			if(m_RegAlpha == 0.0) {
				return -2.0 * ret;
			} else {
				return -2.0 * (ret + m_RegAlpha * Math.abs(weight));
			}
		}
	}
	
	private double calcGain(double sumGrad, double sumHess, double testGrad, double testHess) {
		double weight = calcWeight(sumGrad, sumHess);
		double ret = testGrad * weight + 0.5 * (testHess + m_RegLambda) * sqr(weight);
		if(m_RegAlpha == 0.0) {
			return -2.0 * ret;
		} else {
			return -2.0 * (ret + m_RegAlpha * Math.abs(weight));
		}
	}
	
	private double calcGainGivenWeight(double sumGrad, double sumHess, double weight) {
		return -(2.0 * sumGrad * weight + (sumHess + m_RegLambda) * sqr(weight));
	}
	
	private double sqr(double x) {
		return x * x;
	}*/

	public double classifyInstance(Instance inst) {
		if (m_Attribute == null || leaf == true) {
			return m_ClassValue;
		} else {
			if (inst.value(m_Attribute.index()) < m_splitValue) {
				return m_Successors[0].classifyInstance(inst);
			} else {
				return m_Successors[1].classifyInstance(inst);
			}
		}
	}
	
	public double[] distributionForInstance(Instance inst) {
		if (m_Attribute == null || leaf == true) {
			return m_Distribution;
		} else {
			if (inst.value(m_Attribute.index()) < m_splitValue) {
				return m_Successors[0].distributionForInstance(inst);
			} else {
				return m_Successors[1].distributionForInstance(inst);
			}
		}
	}

	/**
	 * Prints the decision tree using the private toString method from below.
	 * 
	 * @return a textual description of the classifier
	 */
	@Override
	public String toString() {

		if ((m_Distribution == null) && (m_Successors == null)) {
			return "DPC: No model built yet.";
		}
		return "XGBTree\n" + toString(0);
	}

	/**
	 * Outputs a tree at a certain level.
	 * 
	 * @param level
	 *            the level at which the tree is to be printed
	 * @return the tree as string at the given level
	 */
	private String toString(int level) {

		StringBuffer text = new StringBuffer();

		if (m_Attribute == null) {
			if (Utils.isMissingValue(m_ClassValue)) {
				text.append(": null");
			} else {
				text.append(": " + m_ClassValue);
			}
		} else {

			text.append("\n");
			for (int i = 0; i < level; i++) {
				text.append("|  ");
			}
			text.append(m_Attribute.name() + " < " + Utils.doubleToString(m_splitValue, 3));
			if (m_Successors[0] != null)
				text.append(m_Successors[0].toString(level + 1));
			text.append("\n");
			for (int i = 0; i < level; i++) {
				text.append("|  ");
			}
			text.append(m_Attribute.name() + " >= " + Utils.doubleToString(m_splitValue, 3));
			if (m_Successors[1] != null)
				text.append(m_Successors[1].toString(level + 1));

		}
		return text.toString();
	}

	public double sum(double[] array) {
		double total = 0.0;
		for (int i = 0; i < array.length; i++) {
			total += array[i];
		}
		return total;
	}

	public static void main(String[] args) {
		runClassifier(new BackfitTree(), args);
	}
}
