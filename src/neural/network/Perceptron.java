/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neural.network;

import java.util.ArrayList;
import java.util.Scanner;

/**
 *
 * @author Herbert
 */
public class Perceptron {
    
    public static Neurone neurone;
    
    public static void setWeights() {
        if(neurone.getErrorRate() != 0){
            for(int i=0; i<neurone.getnFeatures(); i++){
                double w = neurone.getInWeight().get(i) + neurone.getLearningRate() * neurone.getErrorRate() * neurone.getInFeatures().get(i);
                neurone.getInWeight().set(i, w);
            }
            neurone.setBiasWeight(neurone.getBiasWeight() + neurone.getLearningRate() * neurone.getErrorRate() * neurone.getBias());
        }
    }

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        Scanner sc = new Scanner(System.in);
        System.out.println("Quantos atributos tem cada instância?");
        int nFeatures = Integer.parseInt(sc.nextLine());        
        double learningRate = 0.3;        
        Function f = new SigmoidalFunction();
        
        neurone = new Neurone(nFeatures, learningRate, f);
        
        // TESTE DA FUNÇÂO DE ATIVAÇÂO
        /*for (int i=-9; i<10; i++){
            System.out.println(i+" "+f.calculate((double)i));
        }*/
        
        // Carregar Estrutura de Dados
        ArrayList<ArrayList<Double>> sample = new ArrayList<>();
        ArrayList<Double> classification = new ArrayList<>();
        System.out.println("Entre com as instâncias. Digite 'FIM' quando terminar");
        String line = sc.nextLine();
        while(line.compareTo("FIM")!=0){
            String[] features = line.split(" ");
            ArrayList<Double> instance = new ArrayList<>();
            for(int i=0; i<features.length-1; i++){
                instance.add(Double.parseDouble(features[i]));
            }
            sample.add(instance);
            classification.add(Double.parseDouble(features[features.length-1]));
            line = sc.nextLine();
        }
        
        // Treinamento
        int s = 0;
        double erro = 1.0;
        long init = System.currentTimeMillis();
        
        while(erro > 0.01 || s != 0){   
            if(s == 0){
                erro = 0.0;
            }
            if(neurone.setInstance(sample.get(s), classification.get(s))){
                neurone.proccessInstance();
                setWeights();
                erro += Math.abs(neurone.getErrorRate());
            }
            
            s = (s+1) % sample.size();
        }
        
        long end = System.currentTimeMillis();
        System.out.println("\n\nTREINAMENTO CONCLUIDO EM "+(end-init)+" MILISSEGUNDOS\n\n");
        
        // Teste
        System.out.println("Entre com uma instância para teste");
        line = sc.nextLine();
        while(line.compareTo("exit")!=0){
            String[] features = line.split(" ");            
            ArrayList<Double> instance = new ArrayList<>();
            for(int i=0; i<features.length; i++){
                instance.add(Double.parseDouble(features[i]));
            }
            
                double answer = neurone.classifyInstance(instance);
                if(answer > 0.5){
                    System.out.println("Saida -> 1");
                } else {
                    System.out.println("Saida -> 0");
                }                
            
            line = sc.nextLine();
        }
        
    }
    
}
