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
public class LimiarFunction implements Function{

    @Override
    public double calculate(double value) {
        if(value <= 0.0){
            return 0.0;
        }
        else{
            return 1.0;
        }
    }

    @Override
    public double derivate(double value) {
        return calculate(value);
    }
    
}
