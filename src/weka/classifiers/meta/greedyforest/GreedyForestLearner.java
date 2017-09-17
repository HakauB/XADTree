package weka.classifiers.meta.greedyforest;

import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public class GreedyForestLearner extends AbstractClassifier {

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(data);
		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

	}

}
