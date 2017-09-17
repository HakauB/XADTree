package weka.classifiers.meta.adtree;

import weka.core.Instance;
import weka.core.Instances;

public abstract class SplitterNode {
	
	public int orderAdded;
	
	public abstract int getNumOfBranches();
	
	public abstract int branchInstanceGoesDown(Instance i);
	
	public abstract ReferenceInstances instancesDownBranch(int branch, Instances sourceInstances);
	
	public abstract boolean equalTo(SplitterNode compare);
	
	public abstract void setChildForBranch(int branchNum, PredictionNode childPredictor);
	
	public abstract PredictionNode getChildForBranch(int branchNum);
	
	public abstract Object clone();
	
	public abstract String attributeString(Instances dataset);
	
	public abstract String comparisonString(int branchNum, Instances dataset);
}
