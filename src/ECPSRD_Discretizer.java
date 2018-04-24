package keel.Algorithms.Discretizers.ECPSRD;

/**
 * <p>Title: ECPSD_D</p>
 * 
 * <p>Description: It discretize the train data set using the cut points selected for each attribute
 * obtained from the CHC multivariate algorithm. It makes a faster convergence. </p>
 *
 *
 * <p>Company: KEEL </p>
 *
 * @author Written by Sergio Ramírez (University of Granada) 10/01/2014
 * @version 1.0
 */

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;

import org.core.Files;
import org.core.Randomize;

public class ECPSRD_Discretizer {
    private myDataset train, test;
    private String outputTr, outputTst, outputTxt;
    
    // Algorithm's parameters
    private long seed;
    private int nEvaluationsCHC;  // Number of evaluations for the CHC procedure
    private int populationCHCSize; // Population size for the CHC procedure
    private double rCHC; // Percentage of Change in Restart for the CHC procedure
    private double alpha_fitness; // Alpha for the fitness function
    private double beta_fitness; // Beta for the fitness function
    private double pEvaluationsForReduction; // Percentage evaluations to apply a reduction
    private double pReduction; // Percentile of genes to reduce in each stage
    private double prob0to1Rec; // Probability of change in recombination (1 to 0)
	private double prob0to1Div; // Probability of change in divergence (1 to 0)
    
    private double [][] borderlineCutPoints; // All possible cut points according to the data that can separate classes
    private double [][] cutPoints; // Obtained discretization    
    
    /**
     * Default constructor
     */
    public ECPSRD_Discretizer () {}
    
    /**
     * It reads the data from the input files (training and test) and parse all the parameters
     * from the parameters array.
     * 
     * @param parameters parseParameters It contains the input files, output files and parameters
     */
    public ECPSRD_Discretizer (parseParameters parameters) {

        train = new myDataset();
        test = new myDataset();
        
        try {
            System.out.println("\nReading the training set: " +
                               parameters.getTrainingInputFile());
            train.readClassificationSet(parameters.getTrainingInputFile(), true);
            System.out.println("\nReading the test set: " +
                               parameters.getTestInputFile());
            test.readClassificationSet(parameters.getTestInputFile(), false);
        } catch (IOException e) {
            System.err.println("There was a problem while reading the input data-sets: " + e);
            System.exit(-1);
        }

        outputTr = parameters.getTrainingOutputFile();
        outputTst = parameters.getTestOutputFile();

        outputTxt = parameters.getOutputFile(0);
        
        // Now we parse the parameters, for example:
        seed = Long.parseLong(parameters.getParameter(0));
        
        // Number of evaluations for the CHC procedure
        nEvaluationsCHC = Integer.parseInt(parameters.getParameter(1));

        // Population size for the CHC procedure
        populationCHCSize = Integer.parseInt(parameters.getParameter(2));
        
        // Percentage of Change in Restart for the CHC procedure
        rCHC = Double.parseDouble(parameters.getParameter(3));
        
        // Alpha for the fitness function
        alpha_fitness = Double.parseDouble(parameters.getParameter(4));
        
        // Beta for the fitness function
        beta_fitness = Double.parseDouble(parameters.getParameter(5));
        
        // Getting the probability of change bits
        prob0to1Rec = Double.parseDouble(parameters.getParameter(6));
        prob0to1Div = Double.parseDouble(parameters.getParameter(7));
        
        // Percentage evaluations to apply a reduction
        pEvaluationsForReduction = Double.parseDouble(parameters.getParameter(8));
        
        // Percentile of genes to reduce in each stage
        pReduction = Double.parseDouble(parameters.getParameter(9));
    }
    
    /**
     * It launches the algorithm
     */
    public void execute() {
    	int [] partitions;
    	int index = 0;
    	
    	// We do here the algorithm's operations
        Randomize.setSeed(seed);
        
        // Obtain the borderline cut points
        computeBorderlineCutPoints (train);
        
        // Select the useful cut points with a CHC algorithm
        ECPSRD algorithm = new ECPSRD (seed, train, borderlineCutPoints, nEvaluationsCHC, populationCHCSize, rCHC, 
        		alpha_fitness, beta_fitness,  prob0to1Rec, prob0to1Div, pEvaluationsForReduction, pReduction);
        Result result = algorithm.runCHC();
        boolean[] best_chr = result.getBest();
        double[][] selected_cut_points = result.getCutPoints();
        
        // Copy the results to the cutpoints matrix
        partitions = new int [train.getnInputs()];
        cutPoints = new double[train.getnInputs()][];
        
        for (int i=0; i<train.getnInputs(); i++) {
        	if (selected_cut_points[i] == null) {
        		cutPoints[i] = null;
        		partitions[i] = 0;
        	} else {
            	for (int j=index; j<index+selected_cut_points[i].length; j++) {
            		if (best_chr[j]) {
            			partitions[i]++;
            		}
            	}
        		index += selected_cut_points[i].length;            	
        	}
        }
        
        index = 0;
        for (int i=0; i<train.getnInputs(); i++) {
        	if (selected_cut_points[i] != null) {
        		cutPoints[i] = new double[partitions[i]];
        		
        		int added = 0;
        		for (int j=index; j<index+selected_cut_points[i].length; j++) {
            		if (best_chr[j]) {
            			cutPoints[i][added] = selected_cut_points[i][j-index];
            			added++;
            		}
            	}
        		index += selected_cut_points[i].length;
        	}
        }
        
        // Print the results to the files
        doOutput (train, outputTr);
        doOutput (test, outputTst);
        printCutPoints (outputTxt);
    }
    
    /**
     * Obtains all possible cut points according to the data that can separate classes
     * 
     * @param data	Data-set containing the data we are going to use to obtain the cut points
     */
    private void computeBorderlineCutPoints (myDataset data) {
    	borderlineCutPoints = new double[data.getnInputs()][];
    	ArrayList <Pair> data_by_attribute = new ArrayList <Pair> (data.getnData());
    	ArrayList <Double> border_cut_points = new ArrayList <Double> (data.getnData());
		
    	for (int i=0; i<data.getnInputs(); i++) {
    		if (data.getTipo(i) == myDataset.NOMINAL) {
    			borderlineCutPoints[i] = null;
    		}
    		else {
    			data_by_attribute.clear();
    			border_cut_points.clear();
    	    	
    			// Sort the attribute according to the data values
    			for (int j=0; j<data.getnData(); j++) {
    				data_by_attribute.add(new Pair (data.getOutputAsInteger(j), data.getExample(j)[i]));
    			}
    			Collections.sort(data_by_attribute);
    			
    			// Search for cut points values
    			double valueAnt = ((Pair)(data_by_attribute.get(0))).getProportion();
    			int classAnt = ((Pair)(data_by_attribute.get(0))).getClas();
    			
    			for (int j=1; j<data_by_attribute.size(); j++) {
    				Pair current_pair = (Pair)(data_by_attribute.get(j));
    				double val = current_pair.getProportion();
    				int clas = current_pair.getClas();
    				if (val != valueAnt) {
    					if (clas != classAnt) {
    						boolean found = false;
    						double dif_val = 0.0;
    						for (int k=j-1; k>=0 && !found; k--) {
    							dif_val = ((Pair)(data_by_attribute.get(k))).getProportion();
    							if (dif_val != val) {
    								found = true;
    							}
    						}
    						
    						if (found) {
    							border_cut_points.add((val+dif_val)/2.0);
    						}
    						
    						valueAnt = val;
    						classAnt = clas;
    					}
    				}
    			}
    			
    			// Add the cutpoints to the borderline cut points matrix
    			if (border_cut_points.size() > 0) {
            		borderlineCutPoints[i] = new double[border_cut_points.size()];
            		for (int j=0; j<border_cut_points.size(); j++) {
            			borderlineCutPoints[i][j] = border_cut_points.get(j);
            		}   				
    			}
    			else {
    				borderlineCutPoints[i] = null;
    			}
    		}
    	}
    	
    }
    
    /**
     * Obtains the discretized value of a real data for an attribute considering the cut points vector
     * 
     * @param attribute	Position of the attribute that is associated to the given value
     * @param value	Real value we want to discretize according to the considered cutpoints
     * @return	the integer value associated to the discretization done
     */
	private int discretize (int attribute, double value) {
		if (cutPoints[attribute] == null) 
			return 0;
		
		for(int i=0;i<cutPoints[attribute].length;i++)
			if (value<cutPoints[attribute][i]) 
				return i;
		
		return cutPoints[attribute].length;
	}
    
    /**
     * It generates the output file from a given dataset and stores it in a file
     * 
     * @param dataset myDataset input dataset
     * @param filename String the name of the file
     */
    private void doOutput (myDataset dataset, String filename) {
      double [] sample;    	
      String output = new String("");
      String [] varNames = dataset.varNames();
      output = dataset.copyRelationName(); // We insert the relation line in the output file
      
      // For each parameter, copy its header
      for (int i=0; i < dataset.getnInputs(); i++) {
    	  if (dataset.getTipo(i) != myDataset.NOMINAL) {
    		  output += "@attribute " + varNames[i] + " {";
    		  if (cutPoints[i] != null) {
    			  for (int j=0; j<=cutPoints[i].length; j++) {
    				  output += j;
    				  if (j<cutPoints[i].length)
    					  output += ", ";
    			  }
    		  }
    		  else {
    			  output += "0";
    		  }
    		  output += "}\n";
    	  }
    	  else {
    		  output += dataset.copyAttribute (i);
    	  }
      }
      
      // Finally, we copy the output attribute and the end of the header information
      output += dataset.copyOutput();
      output += dataset.copyEndOfHeader();
      
      // Now we copy the data
      for (int i = 0; i < dataset.getnData(); i++) {
    	  sample = dataset.getExample(i);
    	  
    	  // Copy the attribute values
    	  for (int j=0; j<dataset.getnInputs(); j++) {
    		  if (dataset.isMissing(i, j)) {
    			  output += "?";
    		  }
    		  else {
    			  if (dataset.getTipo(j) == myDataset.NOMINAL) {
    				  output += dataset.getNominalValueAttribute (j, (int)Math.round(sample[j]));
    	    	  }
    	    	  else if ((dataset.getTipo(j) == myDataset.INTEGER)||(dataset.getTipo(j) == myDataset.REAL)) {
    				  output += discretize(j, sample[j]);
    	    	  }
    	    	  else {
    	    		  System.err.println("The current attribute is not a correct one since it has a no correct data type");
    	    		  System.exit(-1);
    	    	  }
    		  }
    		  output += ",";
    	  }
    	  
    	  // Copy the output value
    	  output = output + dataset.getOutputAsString(i) + "\n";
      }
      
      Files.writeFile(filename, output);
    }  

    /**
     * It generates an output file associated to the given dataset which contains the cut points
     * selected and used to discretize its real attributes
     * 
     * @param filename String the name of the file
     */
	private void printCutPoints (String filename) {
		String output = new String("");
		String [] varNames = train.varNames();
		int total_points = 0;
		
		for (int i=0; i<train.getnInputs(); i++) {
			if (train.getTipo(i) != myDataset.NOMINAL) {
				if (cutPoints[i] != null) {
					total_points += cutPoints[i].length;
					for (int j=0; j< cutPoints[i].length; j++) {
						output = output + "Cut point " + j + " of attribute " + i + " (" + varNames[i] + "): " + cutPoints[i][j] + "\n";
					}
					output = output + "Number of cut points of attribute " + i + " (" + varNames[i] + "): " + cutPoints[i].length + "\n";
				} else {
					output = output + "Number of cut points of attribute " + i + " (" + varNames[i] + "): 0\n";
				}				
			} else {
				output = output + "Attribute i (" + varNames[i] + ") is a nominal attribute so it does not have any cut points\n";
			}
		}
		
		output = output + "Total of cut points: " + total_points + "\n";
		
		
		//System.out.println(output);
		Files.writeFile(filename, output);
	}
}
