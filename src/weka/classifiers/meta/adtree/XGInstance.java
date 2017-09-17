package weka.classifiers.meta.adtree;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Utils;

public class XGInstance extends DenseInstance {

    
    public double[] predictionVector;
    public double[] transformedPredictionVector;
    public double[] gradientVector;
    public double[] hessianVector;
    
    private int m_numOfClasses;
    
    public XGInstance(Instance instance) {
      
      super(instance);
      
      setDataset(instance.dataset()); // preserve dataset
      
      m_numOfClasses = instance.numClasses();
      
      // set up vectors
      predictionVector = new double[m_numOfClasses];
      transformedPredictionVector = new double[m_numOfClasses];
      gradientVector = new double[m_numOfClasses];
      hessianVector = new double[m_numOfClasses];
    }
    
    public double[] getGradientVec() {
    	return gradientVector;
    }
    
    public double[] getHessianVec() {
    	return hessianVector;
    }
    
    public double[] getPredictionVec() {
    	return predictionVector;
    }
    
    public void setPredictionVec(double[] preds) {
    	predictionVector = preds;
    }
    
    public void softmaxUpdate() {
    	transformedPredictionVector = new double[m_numOfClasses];
    	double sum = 0.0;
    	for(int i = 0; i < transformedPredictionVector.length; i++) {
    		transformedPredictionVector[i] = Math.exp(predictionVector[i]);
    		sum += transformedPredictionVector[i];
    	}
    	
    	for(int i = 0; i < gradientVector.length; i++) {
    		gradientVector[i] = transformedPredictionVector[i] / sum;
    	}
    	
    	for(int i = 0; i < hessianVector.length; i++) {
    		hessianVector[i] = 1.0;
    	}
    }
    
    
    @Override
    public Object copy() {
    	XGInstance copy = new XGInstance((Instance) super.copy());
      
      System.arraycopy(predictionVector, 0, copy.predictionVector, 0, predictionVector.length);
      System.arraycopy(gradientVector, 0, copy.gradientVector, 0, gradientVector.length);
      System.arraycopy(hessianVector, 0, copy.hessianVector, 0, hessianVector.length);
      
      return copy;
    }
    
    @Override
    public String toString() {
      
      StringBuffer text = new StringBuffer();
      
      text.append(") P(");
      for (int i = 0; i < predictionVector.length; i++) {
        text.append(Utils.doubleToString(predictionVector[i], 3));
        if (i < predictionVector.length - 1) {
          text.append(",");
        }
      }
      text.append(" * G(");
      for (int i = 0; i < gradientVector.length; i++) {
        text.append(Utils.doubleToString(gradientVector[i], 3));
        if (i < gradientVector.length - 1) {
          text.append(",");
        }
      }
      text.append(") H(");
      for (int i = 0; i < hessianVector.length; i++) {
        text.append(Utils.doubleToString(hessianVector[i], 3));
        if (i < hessianVector.length - 1) {
          text.append(",");
        }
      }
      text.append(")");
      return super.toString() + text.toString();
      
    }
  }
