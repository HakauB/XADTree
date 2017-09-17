package weka.classifiers.meta;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.classifiers.meta.adtree.PredictionNode;
import weka.classifiers.meta.adtree.ReferenceInstances;
import weka.classifiers.meta.adtree.Splitter;
import weka.classifiers.meta.adtree.TwoWayNominalSplit;
import weka.classifiers.meta.adtree.TwoWayNumericSplit;
import weka.classifiers.meta.xgboost.ObjectiveFunctions;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class XADTree extends ADTree {
	
	private Attribute m_ClassAttribute;
	
	Attribute m_PredictionAttribute;
	Attribute m_GradientAttribute;
	Attribute m_HessianAttribute;

	private double m_RegAlpha = 0;
	private double m_RegLambda = 1;
	private double m_MaxDeltaStep = 0;
	private double m_MinChildWeight = 1.0;
	
	private double m_LearningRate = 1.0;
	

	private int m_NumSampled = 6;
	
	@Override
	public void buildClassifier(Instances data) throws Exception {

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();
		
		// Add information columns
		data.insertAttributeAt(new Attribute("prediction"), data.numAttributes());
		data.insertAttributeAt(new Attribute("gradient"), data.numAttributes());
		data.insertAttributeAt(new Attribute("hessian"), data.numAttributes());
		m_ClassAttribute = data.attribute(data.numAttributes() - 4);
		m_PredictionAttribute = data.attribute(data.numAttributes() - 3);
		m_GradientAttribute = data.attribute(data.numAttributes() - 2);
		m_HessianAttribute = data.attribute(data.numAttributes() - 1);
		for (int i = 0; i < data.numInstances(); i++) {
			data.instance(i).setValue(m_PredictionAttribute, 0.5);
			data.instance(i).setValue(m_GradientAttribute, 0.0);
			data.instance(i).setValue(m_HessianAttribute, 0.0);
		}

		// set up the tree
		initClassifier(data);

		// build the tree
		for (int T = 0; T < m_boostingIterations; T++) {
			boost();
		}
		
		for (int T = 0; T < 1; T++) {
			//backfit();
		}
		
		// Not too sure if this works properly
		weightCorrect();

		// clean up if desired
		if (!m_saveInstanceData)
			done();
	}
	
	public void backfit() throws Exception {
		backfit(m_root, 1, m_trainInstances);
	}
	
	public void backfit(PredictionNode currentNode, int level, Instances instances) throws Exception {
		
		for (Enumeration e = currentNode.children(); e.hasMoreElements();) {
			Splitter split = (Splitter) e.nextElement();
			
			for (int j = 0; j < split.getNumOfBranches(); j++) {
				Instances branchedInstances = split.instancesDownBranch(j, instances);
				updateWeights(branchedInstances, -split.getChildForBranch(j).getValue());
			}


            double splitPoint = Double.MAX_VALUE;
            double gain = Double.NEGATIVE_INFINITY;
            int att = 0;

            for(int attIndex : m_numericAttIndices) {
                double gradientSum = instances.attributeStats(instances.numAttributes() - 2).numericStats.sum;
                double hessianSum = instances.attributeStats(instances.numAttributes() - 1).numericStats.sum;
                double gradLeftSum = 0;
                double gradRightSum = gradientSum;
                double hessLeftSum = 0;
                double hessRightSum = hessianSum;

                // sort instances
                instances.sort(attIndex);

                for (int i = 0; i < instances.numInstances() - 1; i++) {

                    gradLeftSum += instances.instance(i).value(m_GradientAttribute);
                    gradRightSum = gradientSum - gradLeftSum;

                    hessLeftSum += instances.instance(i).value(m_HessianAttribute);
                    hessRightSum = hessianSum - hessLeftSum;

                    double newScore = calcGain(gradLeftSum, hessLeftSum) + calcGain(gradRightSum, hessRightSum);// - calcGain(gradientSum, hessianSum);
                    if (instances.instance(i).value(attIndex) != instances.instance(i + 1).value(attIndex)) {
                        if (newScore > gain) {
                            gain = newScore;
                            splitPoint = (instances.instance(i).value(attIndex) + instances.instance(i + 1).value(attIndex)) / 2;
                            att = attIndex;
                        }
                    }
                }
            }

            split.setAttIndex(att);
            split.setSplitPoint(splitPoint);

			for (int j = 0; j < split.getNumOfBranches(); j++) {
				
				Instances branchInstances = split.instancesDownBranch(j, instances);
				double predictionValue = calcPredictionValue(branchInstances);
				split.getChildForBranch(j).setValue(predictionValue);
				updateWeights(branchInstances, predictionValue);
			}
		}

		for(Enumeration e = currentNode.children(); e.hasMoreElements();) {
		    Splitter split = (Splitter) e.nextElement();
		    for(int j = 0; j < split.getNumOfBranches(); j++) {
                Instances branchInstances = split.instancesDownBranch(j, instances);
                backfit(split.getChildForBranch(j), (level + 1), branchInstances);
            }
        }
	}
	
	public void weightCorrect() {
		weightCorrect(m_root, 1, m_trainInstances);
	}
	
	
	public void weightCorrect(PredictionNode currentNode, int level, Instances instances) {

		for (Enumeration e = currentNode.children(); e.hasMoreElements();) {
			Splitter split = (Splitter) e.nextElement();

			for (int j = 0; j < split.getNumOfBranches(); j++) {
			    Instances branchInstances = split.instancesDownBranch(j, instances);
			    double predValue = calcPredictionValue(branchInstances);
			    PredictionNode predictionNode = split.getChildForBranch(j);
			    predictionNode.setValue(predictionNode.getValue() + predValue);


				/*Instances posInstances = split.instancesDownBranch(j, instances);
				Instances negInstances = split.instancesDownBranch(j, instances);
				double predictionValue = calcPredictionValue(posInstances, negInstances);
				PredictionNode pred = split.getChildForBranch(j);
				pred.setValue(pred.getValue() + predictionValue);
				Instances curr = new Instances(posInstances);
				for(int k = 0; k < negInstances.numInstances(); k++) {
					curr.add(negInstances.instance(k));
				}
				weightCorrect(pred, (level + 1), curr);*/
			}
		}
		for(Enumeration e = currentNode.children(); e.hasMoreElements();) {
            Splitter split = (Splitter) e.nextElement();
            for (int j = 0; j < split.getNumOfBranches(); j++) {
                Instances branchInstances = split.instancesDownBranch(j, instances);
                weightCorrect(split.getChildForBranch(j), (level + 1), branchInstances);
            }
        }
	}
	
	
	@Override
	public void initClassifier(Instances instances) throws Exception {

		// clear stats
		m_nodesExpanded = 0;
		m_examplesCounted = 0;
		m_lastAddedSplitNum = 0;

		// prepare the random generator
		m_random = new Random(m_randomSeed);

		// create training set
		m_trainInstances = new Instances(instances);

		// create positive/negative subsets
		m_posTrainInstances = new ReferenceInstances(m_trainInstances, m_trainInstances.numInstances());
		m_negTrainInstances = new ReferenceInstances(m_trainInstances, m_trainInstances.numInstances());
		for (Enumeration e = m_trainInstances.enumerateInstances(); e.hasMoreElements();) {
			Instance inst = (Instance) e.nextElement();
			if ((int) inst.classValue() == 0)
				m_negTrainInstances.addReference(inst); // belongs in negative
														// class
			else
				m_posTrainInstances.addReference(inst); // belongs in positive
														// class
		}
		m_posTrainInstances.compactify();
		m_negTrainInstances.compactify();

		// create the root prediction node
		double rootPredictionValue = calcPredictionValue(m_posTrainInstances, m_negTrainInstances);
		m_root = new PredictionNode(rootPredictionValue);

		// pre-adjust weights
		updateWeights(m_posTrainInstances, m_negTrainInstances, rootPredictionValue);

		// pre-calculate what we can
		generateAttributeIndicesSingle();
	}
	
	@Override
	protected void updateWeights(Instances posInstances, Instances negInstances, double predictionValue) {

		for (Enumeration e = posInstances.enumerateInstances(); e.hasMoreElements();) {
			Instance inst = (Instance) e.nextElement();
			inst.setValue(m_PredictionAttribute, inst.value(m_PredictionAttribute) + predictionValue);
		}
		for (Enumeration e = negInstances.enumerateInstances(); e.hasMoreElements();) {
			Instance inst = (Instance) e.nextElement();
			inst.setValue(m_PredictionAttribute, inst.value(m_PredictionAttribute) + predictionValue);
		}
		
		ObjectiveFunctions f = new ObjectiveFunctions(posInstances, m_PredictionAttribute, m_GradientAttribute, m_HessianAttribute, "linloss");
		ObjectiveFunctions g = new ObjectiveFunctions(negInstances, m_PredictionAttribute, m_GradientAttribute, m_HessianAttribute, "linloss");
	}

    protected void updateWeights(Instances instances, double predictionValue) {
        for (Enumeration e = instances.enumerateInstances(); e.hasMoreElements();) {
            Instance inst = (Instance) e.nextElement();
            inst.setValue(m_PredictionAttribute, inst.value(m_PredictionAttribute) + predictionValue);
            ObjectiveFunctions f = new ObjectiveFunctions(instances, m_PredictionAttribute, m_GradientAttribute, m_HessianAttribute, "linloss");
        }
    }

	
	private double weightSum(Instances data) {
		/*double sum = 0;
		for(int i = 0; i < data.numInstances(); i++) {
			Instance inst = data.instance(i);
			sum += -inst.value(m_GradientAttribute) / (inst.value(m_HessianAttribute) + m_RegLambda);
		}*/
		double gradSum = data.attributeStats(data.numAttributes() - 2).numericStats.sum;
		double hessSum = data.attributeStats(data.numAttributes() - 1).numericStats.sum;
		return -gradSum / (hessSum + m_RegLambda);
	}
	
	protected double calcPredictionValue(Instances posInstances, Instances negInstances) {

		
		Instances total = new Instances(posInstances);
		for(int i = 0; i < negInstances.numInstances(); i++) {
			total.add(negInstances.instance(i));
		}
		
		double gradSum = total.attributeStats(posInstances.numAttributes() - 2).numericStats.sum;
		double hessSum = total.attributeStats(posInstances.numAttributes() - 1).numericStats.sum;

		return m_LearningRate * calcWeight(gradSum, hessSum);
	}

    protected double calcPredictionValue(Instances instances) {
        double gradSum = instances.attributeStats(instances.numAttributes() - 2).numericStats.sum;
        double hessSum = instances.attributeStats(instances.numAttributes() - 1).numericStats.sum;

        return m_LearningRate * calcWeight(gradSum, hessSum);
    }

	@Override
	protected void generateAttributeIndicesSingle() {

		// insert indices into vectors
		ArrayList<Integer> nominalIndices = new ArrayList<Integer>();
		ArrayList<Integer> numericIndices = new ArrayList<Integer>();

		for (int i = 0; i < m_trainInstances.numAttributes() - 4; i++) {
				if (m_trainInstances.attribute(i).isNumeric())
					numericIndices.add(new Integer(i));
				else
					nominalIndices.add(new Integer(i));
		}

		// create nominal array
		m_nominalAttIndices = new int[nominalIndices.size()];
		for (int i = 0; i < nominalIndices.size(); i++)
			m_nominalAttIndices[i] = ((Integer) nominalIndices.get(i)).intValue();

		// create numeric array
		m_numericAttIndices = new int[numericIndices.size()];
		for (int i = 0; i < numericIndices.size(); i++)
			m_numericAttIndices[i] = ((Integer) numericIndices.get(i)).intValue();
	}
	
	public void boost() throws Exception {

		if (m_trainInstances == null || m_trainInstances.numInstances() == 0)
			throw new Exception("Trying to boost with no training data");

		// perform the search
		searchForBestTestSingle();
		
		if (m_search_bestSplitter == null)
			return; // handle empty instances

		// create the new nodes for the tree, updating the weights
		for (int i = 0; i < 2; i++) {
			Instances posInstances = m_search_bestSplitter.instancesDownBranch(i, m_search_bestPathPosInstances);
			Instances negInstances = m_search_bestSplitter.instancesDownBranch(i, m_search_bestPathNegInstances);
			double predictionValue = calcPredictionValue(posInstances, negInstances);
			PredictionNode newPredictor = new PredictionNode(predictionValue);
			updateWeights(posInstances, negInstances, predictionValue);
			m_search_bestSplitter.setChildForBranch(i, newPredictor);
		}

		// insert the new nodes
		m_search_bestInsertionNode.addChild((Splitter) m_search_bestSplitter, this);

		// free memory
		m_search_bestPathPosInstances = null;
		m_search_bestPathNegInstances = null;
		m_search_bestSplitter = null;
	}
	
	
	private void searchForBestTestSingle() throws Exception {

		// keep track of total weight for efficient wRemainder calculations
		//m_trainTotalWeight = weightSum(m_trainInstances);

		m_search_smallestZ = Double.NEGATIVE_INFINITY;
		searchForBestTestSingle(m_root, m_posTrainInstances, m_negTrainInstances);
	}
	
	private void searchForBestTestSingle(PredictionNode currentNode, Instances posInstances, Instances negInstances)
			throws Exception {

		//System.out.println(posInstances.numInstances() + " ," + negInstances.numInstances());
		
		// don't investigate pure or empty nodes any further
		if (posInstances.numInstances() == 0 || negInstances.numInstances() == 0)
			//return;

		// do z-pure cutoff
		//if (calcZpure(posInstances, negInstances) >= m_search_smallestZ)
		//	return;

		// keep stats
		m_nodesExpanded++;
		m_examplesCounted += posInstances.numInstances() + negInstances.numInstances();

		
		
		// evaluate dynamic splitters (numeric)
		if (m_numericAttIndices.length > 0) {

			// merge the two sets of instances into one
			Instances allInstances = new Instances(posInstances);
			for (Enumeration e = negInstances.enumerateInstances(); e.hasMoreElements();)
				allInstances.add((Instance) e.nextElement());

			// use method of finding the optimal Z split-point
			for (int i = 0; i < m_numericAttIndices.length; i++)
				evaluateNumericSplitSingle(m_numericAttIndices[i], currentNode, posInstances, negInstances,
						allInstances);
		}

		if (currentNode.getChildren().size() == 0)
			return;

		
		
		/*switch (m_searchPath) {
		case SEARCHPATH_ALL:
			goDownAllPathsSingle(currentNode, posInstances, negInstances);
			break;
		case SEARCHPATH_RANDOM:
			goDownRandomPathSingle(currentNode, posInstances, negInstances);
			break;
		}*/
		
		//goDownRandomPathSingle(currentNode, posInstances, negInstances);
		//goDownRandomlySampledBest(currentNode, posInstances, negInstances);
		goDownAllPathsSingle(currentNode, posInstances, negInstances);
		
	}
	
	private void goDownAllPathsSingle(PredictionNode currentNode, Instances posInstances, Instances negInstances)
			throws Exception {

		for (Enumeration e = currentNode.children(); e.hasMoreElements();) {
			Splitter split = (Splitter) e.nextElement();
			for (int i = 0; i < split.getNumOfBranches(); i++)
				searchForBestTestSingle(split.getChildForBranch(i), split.instancesDownBranch(i, posInstances),
						split.instancesDownBranch(i, negInstances));
		}
	}
	
	private void goDownRandomPathSingle(PredictionNode currentNode, Instances posInstances, Instances negInstances)
			throws Exception {

		FastVector children = currentNode.getChildren();
		Splitter split = (Splitter) children.elementAt(getRandom(children.size()));
		int branch = getRandom(split.getNumOfBranches());
		searchForBestTestSingle(split.getChildForBranch(branch), split.instancesDownBranch(branch, posInstances),
				split.instancesDownBranch(branch, negInstances));
	}
	
	private void goDownRandomlySampledBest(PredictionNode currentNode, Instances posInstances, Instances negInstances)
			throws Exception {
		Splitter heaviestSplit = null;
		
		FastVector children = currentNode.getChildren();
		
		Splitter[] sample = new Splitter[m_NumSampled];
		for(int i = 0; i < m_NumSampled; i++) {
			sample[i] = (Splitter) children.elementAt(getRandom(children.size()));
		}
		
		int bestIndex = 0;
		double bestWeight = 0.0;
		
		for(int i = 0; i < m_NumSampled; i++) {
			
			for(int j = 0; j < sample[i].getNumOfBranches(); j++) {
				double weight = (Math.abs(weightSum(sample[i].instancesDownBranch(j, posInstances)))
						+ Math.abs(weightSum(sample[i].instancesDownBranch(j, negInstances)))) / (currentNode.getChildren().size() + 1);
				if (weight > bestWeight) {
					heaviestSplit = sample[i];
					bestIndex = j;
					bestWeight = weight;
				}
			}
		}
		
		
		/*Splitter split = (Splitter) children.elementAt(getRandom(children.size()));
		int branch = getRandom(split.getNumOfBranches());
		searchForBestTestSingle(split.getChildForBranch(branch), split.instancesDownBranch(branch, posInstances),
				split.instancesDownBranch(branch, negInstances));*/
		if (heaviestSplit != null)
			searchForBestTestSingle(heaviestSplit.getChildForBranch(bestIndex),
					heaviestSplit.instancesDownBranch(bestIndex, posInstances),
					heaviestSplit.instancesDownBranch(bestIndex, negInstances));
	}
	
	private void evaluateNumericSplitSingle(int attIndex, PredictionNode currentNode, Instances posInstances,
			Instances negInstances, Instances allInstances) throws Exception {

		double[] splitAndZ = findLowestZNumericSplit(allInstances, attIndex);

		if (splitAndZ[1] > m_search_smallestZ) {
			m_search_smallestZ = splitAndZ[1];
			m_search_bestInsertionNode = currentNode;
			m_search_bestSplitter = new TwoWayNumericSplit(attIndex, splitAndZ[0]);
			m_search_bestPathPosInstances = posInstances;
			m_search_bestPathNegInstances = negInstances;
		}
	}
	
	private double[] findLowestZNumericSplit(Instances instances, int attIndex) throws Exception {

		double splitPoint = Double.MAX_VALUE;
		
		double gain = Double.MIN_VALUE;
		double gradientSum = instances.attributeStats(instances.numAttributes() - 2).numericStats.sum;
		double hessianSum = instances.attributeStats(instances.numAttributes() - 1).numericStats.sum;
		double gradLeftSum = 0;
		double gradRightSum = gradientSum;
		double hessLeftSum = 0;
		double hessRightSum = hessianSum;

		// sort instances
		instances.sort(attIndex);
		

		// make split counts for each possible split and evaluate
		for (int i = 0; i < instances.numInstances() - 1; i++) {

			if(instances.instance(i).isMissing(attIndex)) {
				continue;
			}
			
			gradLeftSum += instances.instance(i).value(m_GradientAttribute);
			gradRightSum = gradientSum - gradLeftSum;

			hessLeftSum += instances.instance(i).value(m_HessianAttribute);
			hessRightSum = hessianSum - hessLeftSum;

			//double newScore = (Math.pow(gradLeftSum, 2.0) / hessLeftSum)
			//		+ (Math.pow(gradRightSum, 2.0) / hessRightSum) - (Math.pow(gradientSum, 2.0) / hessianSum);
			double newScore = calcGain(gradLeftSum, hessLeftSum) + calcGain(gradRightSum, hessRightSum);// - calcGain(gradientSum, hessianSum);
			if (instances.instance(i).value(attIndex) != instances.instance(i + 1).value(attIndex)) {
				if (newScore > gain) {
					gain = newScore;
					splitPoint = (instances.instance(i).value(attIndex) + instances.instance(i + 1).value(attIndex)) / 2;
				}
			}
		}
		
		

		double[] splitAndZ = new double[2];
		splitAndZ[0] = splitPoint;
		splitAndZ[1] = gain;
		return splitAndZ;
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
	
	private double sqr(double x) {
		return x * x;
	}
	
	private double L1Threshold(double weight, double lambda) {
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
	
	private double instWeight(Instance inst) {
		return -inst.value(m_GradientAttribute) / (inst.value(m_HessianAttribute) + m_RegLambda);
	}
	
	public static void main(String[] args) {
		runClassifier(new XADTree(), args);
	}

}
