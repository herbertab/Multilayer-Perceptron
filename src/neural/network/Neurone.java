/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.util.ArrayList;

/**
 *
 * @author Herbert
 */
public class Neurone {
    
    private static final double bias = -1.0;    
    
    private int nFeatures;
    private double biasWeight;
    private double learningRate;
    private double errorRate;
    private double realOutValue;
    private double outValue;
    private ArrayList<Double> inFeatures;
    private double goal;
    private ArrayList<Double> inWeight;
    private Function function;
    
    public Neurone(int nFeatures, double learningRate, Function function){        
        this.nFeatures = nFeatures;
        this.learningRate = learningRate;
        this.biasWeight = Math.random();
        this.errorRate = this.goal = this.outValue = this.realOutValue = this.bias;
        this.inFeatures = new ArrayList<>();
        this.inWeight = new ArrayList<>();
        for(int i=0; i<nFeatures; i++){
            inFeatures.add(0.0);
            inWeight.add(Math.random());
        }        
        this.function = function;
    }
    
    public boolean setInstance(ArrayList<Double> instance, double classification){        
        if(instance.size() == getnFeatures()){
            this.setInFeatures(instance);
            this.setGoal(classification);
            return true;
        }
        else{
            System.out.println("INVALID INSTANCE");
            return false;
        }        
    }
    
    public void proccessInstance(){        
        outValue = 0.0;        
        for(int i=0; i<getnFeatures(); i++){
            outValue += getInFeatures().get(i) * getInWeight().get(i);
        }
        outValue += getBias() * getBiasWeight();
        
        setRealOutValue(getFunction().calculate(outValue));
        
        //System.out.println("Entrada "+inFeatures.get(0)+" e "+inFeatures.get(1)+" Saída "+realOutValue);
        
        this.setErrorRate(this.getGoal() - getRealOutValue());   
        //System.out.println("Meta ->"+goal+" "+"Erro ->"+errorRate);
    }
    
    public double classifyInstance(ArrayList<Double> instance){
        outValue = 0.0;
        if(instance.size() == getnFeatures()){
            this.setInFeatures(instance);            
        }
        else{
            System.out.println("INVALID INSTANCE");
            return -1.0;
        }
        outValue = 0.0;        
        for(int i=0; i<getnFeatures(); i++){
            outValue += getInFeatures().get(i) * getInWeight().get(i);
        }
        outValue += getBias() * getBiasWeight();
        System.out.println("Entrada "+inFeatures.get(0)+" e "+inFeatures.get(1)+" Saída "+getFunction().calculate(outValue));
        return getFunction().calculate(outValue);
    }   
    
    /**
     * @return the bias
     */
    public static double getBias() {
        return bias;
    }

    /**
     * @return the nFeatures
     */
    public int getnFeatures() {
        return nFeatures;
    }

    /**
     * @return the learningRate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * @return the errorRate
     */
    public double getErrorRate() {
        return errorRate;
    }

    /**
     * @return the realOutValue
     */
    public double getRealOutValue() {
        return realOutValue;
    }

    /**
     * @param nFeatures the nFeatures to set
     */
    public void setnFeatures(int nFeatures) {
        this.nFeatures = nFeatures;
    }

    /**
     * @return the biasWeight
     */
    public double getBiasWeight() {
        return biasWeight;
    }

    /**
     * @param biasWeight the biasWeight to set
     */
    public void setBiasWeight(double biasWeight) {
        this.biasWeight = biasWeight;
    }

    /**
     * @param learningRate the learningRate to set
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * @param errorRate the errorRate to set
     */
    public void setErrorRate(double errorRate) {
        this.errorRate = errorRate;
    }

    /**
     * @param realOutValue the realOutValue to set
     */
    public void setRealOutValue(double realOutValue) {
        this.realOutValue = realOutValue;
    }

    /**
     * @return the inFeatures
     */
    public ArrayList<Double> getInFeatures() {
        return inFeatures;
    }

    /**
     * @param inFeatures the inFeatures to set
     */
    public void setInFeatures(ArrayList<Double> inFeatures) {
        this.inFeatures = inFeatures;
    }

    /**
     * @return the goal
     */
    public double getGoal() {
        return goal;
    }

    /**
     * @param goal the goal to set
     */
    public void setGoal(double goal) {
        this.goal = goal;
    }

    /**
     * @return the inWeight
     */
    public ArrayList<Double> getInWeight() {
        return inWeight;
    }

    /**
     * @param inWeight the inWeight to set
     */
    public void setInWeight(ArrayList<Double> inWeight) {
        this.inWeight = inWeight;
    }

    /**
     * @return the function
     */
    public Function getFunction() {
        return function;
    }

    /**
     * @param function the function to set
     */
    public void setFunction(Function function) {
        this.function = function;
    }

    /**
     * @return the outValue
     */
    public double getOutValue() {
        return outValue;
    }

    /**
     * @param outValue the outValue to set
     */
    public void setOutValue(double outValue) {
        this.outValue = outValue;
    }
}
