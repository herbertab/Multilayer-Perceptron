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
public interface Function {
    double calculate(double value);
    double derivate(double value);
}
