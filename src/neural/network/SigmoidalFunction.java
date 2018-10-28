/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

/**
 *
 * @author Herbert
 */
public class SigmoidalFunction implements Function{

    @Override
    public double calculate(double value) {
        
        double e = 2.718281828;
        return 1.0/(1.0 + Math.pow(1.0/e, value));
        
    }
    
    public double derivate(double value){
        double e = 2.718281828;
        return Math.pow(1.0/e, value) / (Math.pow(Math.pow(1.0/e, value)+1.0, 2.0));
        //return value * (1 - value);
    }
    
}
