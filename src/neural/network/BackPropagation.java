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
public class BackPropagation {
    
    public static ArrayList<ArrayList<Neurone>> mlp;
    
    
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        ArrayList<Integer> sizes = new ArrayList<>();
        Function f = new SigmoidalFunction();
        
        // TESTE DA FUNÇÂO DE ATIVAÇÂO
        /*for (int i=-9; i<10; i++){
            System.out.println(i+" "+f.derivate((double)i));
        }*/
        
        System.out.println("Quantos neurônios na camada de entrada?");
        int inLayerLength = Integer.parseInt(sc.nextLine());
        sizes.add(inLayerLength);
                      
        System.out.println("Quantas camadas intermediarias?");
        int nInnerLayers = Integer.parseInt(sc.nextLine());
        int innerLayerLength = 0;
        
        if(nInnerLayers > 0){
            
            for(int k=0; k<nInnerLayers; k++){
                System.out.println("Quantos neurônios na(s) camada(s) intermediárias?");
                innerLayerLength = Integer.parseInt(sc.nextLine());
                sizes.add(innerLayerLength);
            }
        }
        
        System.out.println("Quantos neurônios na camada de saída?");
        int outLayerLength = Integer.parseInt(sc.nextLine());
        sizes.add(outLayerLength);
        
        int layers = 2 + nInnerLayers;
        
        System.out.println("Quantos atributos tem cada instância?");
        int nFeatures = Integer.parseInt(sc.nextLine());
        double learningRate = 0.2;
        
        
        // Inicialização dos neuronios
        mlp = new ArrayList<>();
        for(int i=0; i<layers; i++){
            ArrayList<Neurone> layer = new ArrayList<>();
            for(int j=0; j<sizes.get(i); j++){
                Neurone neurone;
                if(i == 0){
                    neurone = new Neurone(nFeatures, learningRate, f);
                } else {
                    neurone = new Neurone(sizes.get(i-1), learningRate, f);
                }
                layer.add(neurone);
            }
            mlp.add(layer);
        }
        
        // Carregamento da base de dados
        ArrayList<ArrayList<Double>> sample = new ArrayList<>();
        ArrayList<ArrayList<Double>> classification = new ArrayList<>();
        System.out.println("Entre com as instâncias. Digite 'FIM' quando terminar");
        String line = sc.nextLine();
        while(line.compareTo("FIM")!=0){
            String[] features = line.split(" ");
            ArrayList<Double> instance = new ArrayList<>();
            for(int i=0; i<features.length-outLayerLength; i++){
                instance.add(Double.parseDouble(features[i]));
            }
            sample.add(instance);
            ArrayList<Double> c = new ArrayList<>();
            for(int i=nFeatures; i<nFeatures + outLayerLength; i++){
                c.add(Double.parseDouble(features[i]));
            }
            classification.add(c);
            line = sc.nextLine();
        }
        
        // Propagação dos valores da entrada
        int s = 0;
        int acertos = 0;
        double cumulativeError = 5.0;
        long init = System.currentTimeMillis();
        while((cumulativeError > 0.49 && acertos < sample.size()) || s != 0){   
            //System.out.println("-----------------------------------------"+sample.size());
            
            if(s == 0){
                acertos = 0;
                cumulativeError = 0.0;
            }
            
            // submete a instância a cada neuronio da camada de entrada
            for(int i=0; i<mlp.get(0).size(); i++){
                /*if(s == 0){
                    System.out.println("INSTANCIA");
                }*/
                if(mlp.get(0).get(i).setInstance(sample.get(s), 0.0)){                    
                    mlp.get(0).get(i).proccessInstance();
                }
            }
            
            // para cada camada subsequente, submete as saidas da camada anterior como entrada
            for(int j=1; j<mlp.size(); j++){
                for(int i=0; i<mlp.get(j).size(); i++){
                    if(j < mlp.size()-1){
                        if(mlp.get(j).get(i).setInstance(getOutValues(j-1), 0.0)){
                            mlp.get(j).get(i).proccessInstance();
                        }
                    } else {
                        if(mlp.get(j).get(i).setInstance(getOutValues(j-1), classification.get(s).get(i))){
                            mlp.get(j).get(i).proccessInstance();
                        }
                    }
                }
            }
            
            // back propagation (propagacao do erro para tras)
            for(int j=mlp.size()-2; j>=0; j--){
                for(int i=0; i<mlp.get(j).size(); i++){
                    mlp.get(j).get(i).setErrorRate(calculateWeight(i, j+1));
                }
            }
            
            // ajuste dos pesos para cada neurônio
            for(int i=0; i<mlp.size(); i++){    //para cada camada
                for(int j=0; j<mlp.get(i).size(); j++){ //para cada neuronio na camada
                    if(mlp.get(i).get(j).getErrorRate() != 0){ // se erro existir
                        for(int k=0; k<mlp.get(i).get(j).getnFeatures(); k++){ // para cada entrada do neuronio
                            double w = mlp.get(i).get(j).getInWeight().get(k) + 
                                    mlp.get(i).get(j).getLearningRate() * mlp.get(i).get(j).getErrorRate() * 
                                    f.derivate(mlp.get(i).get(j).getOutValue()) * mlp.get(i).get(j).getInFeatures().get(k);
                            mlp.get(i).get(j).getInWeight().set(k, w);
                        }
                        mlp.get(i).get(j).setBiasWeight(mlp.get(i).get(j).getBiasWeight() + 
                                mlp.get(i).get(j).getLearningRate() * mlp.get(i).get(j).getErrorRate() * 
                                        f.derivate(mlp.get(i).get(j).getOutValue()) * mlp.get(i).get(j).getBias());
                    }
                }
            } 
            if(getAcertos() == mlp.get(mlp.size()-1).size()){
                acertos++;
            }
            cumulativeError += getError(); 
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
            
                ArrayList<Double> answer = classify(instance);
                for(int i = 0; i<answer.size(); i++){
                    
                    if(answer.get(i) > 0.5){
                        System.out.println("Saida do Neuronio "+i+" -> 1");
                    } else {
                        System.out.println("Saida do Neuronio "+i+" -> 0");
                    } 
                }
            
            line = sc.nextLine();
        }
        
    }

    private static ArrayList<Double> getOutValues(int i) {
        ArrayList<Double> out = new ArrayList<>();
        for(int j=0; j<mlp.get(i).size(); j++){
            out.add(mlp.get(i).get(j).getRealOutValue());
        }
        return out;
    }

    private static double calculateWeight(int i, int layer) {
        double value = 0.0;
        
        for(int j=0; j<mlp.get(layer).size(); j++){
            value += mlp.get(layer).get(j).getInWeight().get(i) * mlp.get(layer).get(j).getErrorRate();
        }
        
        return value;
    }

    private static double getError() {
        double error = 0.0;
        for(int i=0; i<mlp.get(mlp.size()-1).size(); i++){
            error += Math.abs(mlp.get(mlp.size()-1).get(i).getErrorRate());
        }
        return error;
    }

    private static ArrayList<Double> classify(ArrayList<Double> instance) {
        
        ArrayList<Double> out = new ArrayList<>();
        
        // submete a instância a cada neuronio da camada de entrada
            for(int i=0; i<mlp.get(0).size(); i++){
                if(mlp.get(0).get(i).setInstance(instance, 0.0)){
                    mlp.get(0).get(i).proccessInstance();
                }
            }
            
            // para cada camada subsequente, submete as saidas da camada anterior como entrada
            for(int j=1; j<mlp.size(); j++){
                for(int i=0; i<mlp.get(j).size(); i++){
                    if(j < mlp.size()-1){
                        if(mlp.get(j).get(i).setInstance(getOutValues(j-1), 0.0)){
                            mlp.get(j).get(i).proccessInstance();
                        }
                    } else {
                        if(mlp.get(j).get(i).setInstance(getOutValues(j-1), 0.0)){
                            mlp.get(j).get(i).proccessInstance();
                            out.add(mlp.get(j).get(i).getRealOutValue());
                        }
                    }
                }
            }
        
        return out;
    }

    private static int getAcertos() {
        int count = 0;
        
        for(int i=0; i<mlp.get(mlp.size()-1).size(); i++){
            double saida = mlp.get(mlp.size()-1).get(i).getRealOutValue();
            if(saida > 0.5){
                saida = 1.0;
            }else{
                saida = 0.0;
            }
            if(mlp.get(mlp.size()-1).get(i).getGoal() == saida){
                
                count++;
            }
        }
        //System.out.println("-------------------->>>>>>>>>>>>>>>>>>>>>>>>>>>>>"+count);
        return count;
    }
    
}


































