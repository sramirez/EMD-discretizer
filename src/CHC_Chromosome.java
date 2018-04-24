package keel.Algorithms.Discretizers.ECPSRD;

import org.core.Randomize;




import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.*;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>Title: CHC_RuleBase </p>
 *
 * <p>Description: Chromosome that represents a rule base used in the CHC algorithm </p>
 *
 * <p>Company: KEEL </p>
 *
 * @author Written by Victoria Lopez (University of Granada) 30/04/2011
 * @author Modified by Victoria Lopez (University of Granada) 04/05/2011
 * @version 1.5
 * @since JDK1.5
 */

public class CHC_Chromosome implements Comparable {
	private boolean [] individual; // Boolean array selecting cutpoints from a list of cutpoints
	private boolean n_e; // Indicates whether this chromosome has been evaluated or not
	double fitness;// Fitness associated to the cut points represented by the boolean array
	int n_cutpoints; // Fitness associated to the dataset, it indicates the number of cutpoints selected
	int inconsistencies;
	double perc_err;
	
	/**
     * Default constructor
     */
    public CHC_Chromosome () {
    }
    
    /**
     * Creates a CHC chromosome from another chromosome (copies a chromosome)
     * 
     * @param orig	Original chromosome that is going to be copied
     */
    public CHC_Chromosome (CHC_Chromosome orig) {
    	individual = new boolean [orig.individual.length];
    	
    	for (int i=0; i<orig.individual.length; i++) {
    		individual[i] = orig.individual[i];
    	}
    	
    	n_e = orig.n_e;
    	fitness = orig.fitness;
    	n_cutpoints = orig.n_cutpoints;
    }
    
    /**
     * Creates a random CHC_Chromosome of specified size
     * 
     * @param size	Size of the new chromosome 
     */
    public CHC_Chromosome (int size) {
    	double u;
    	
    	individual = new boolean [size];
    	
    	for (int i=0; i<size; i++) {
    		u = Randomize.Rand();
			if (u < 0.5) {
				individual[i] = false;
			}
			else {
				individual[i] = true;
			}
    	}
    	
    	n_e = true;
    	fitness = 0.0;
    	n_cutpoints = size;
    }

    /**
     * Creates a CHC_Chromosome of specified size with all its elements set to the specified value
     * 
     * @param size	Size of the new chromosome
     * @param value	Value that all elements of the chromosome are going to have 
     */
    public CHC_Chromosome (int size, boolean value) {
    	individual = new boolean [size];
    	
    	for (int i=0; i<size; i++) {
    		individual[i] = value;
    	}
    	
    	n_e = true;
    	fitness = 0.0;
    	n_cutpoints = size;
    }
    
    /**
     * Creates a CHC chromosome from a boolean array representing a chromosome
     * 
     * @param data	boolean array representing a chromosome
     */
    public CHC_Chromosome (boolean data[]) {
    	individual = new boolean [data.length];
    	
    	for (int i=0; i<data.length; i++) {
    		individual[i] = data[i];
    	}
    	
    	n_e = true;
    	fitness = 0.0;
    	n_cutpoints = data.length;
    }
    
    /**
     * Creates a CHC chromosome from another chromosome using the CHC diverge procedure
     * 
     * @param orig	Best chromosome of the population that is going to be used to create another chromosome
     * @param r	R factor of diverge
     */
    public CHC_Chromosome (CHC_Chromosome orig, double r) {
    	individual = new boolean [orig.individual.length];
    	
    	for (int i=0; i<orig.individual.length; i++) {
    		if (Randomize.Rand() < r) {
    			individual[i] = !orig.individual[i];
    		}
    		else {
    			individual[i] = orig.individual[i];
    		}
    	}
    	
    	n_e = true;
    	fitness = 0.0;
    	n_cutpoints = orig.n_cutpoints;
    }

    /**
     * Creates a CHC chromosome from another chromosome using the CHC diverge procedure
     * 
     * @param orig	Best chromosome of the population that is going to be used to create another chromosome
     * @param r	R factor of diverge
     */
    public CHC_Chromosome (CHC_Chromosome orig, double r, double prob0to1Div) {
    	individual = new boolean [orig.individual.length];
    	
    	for (int i=0; i<orig.individual.length; i++) {
    		if (Randomize.Rand() < r) {
    			if (Randomize.Rand() < prob0to1Div) {
    				individual[i] = true;
    			} else {
    				individual[i] = false;
    			}
    		}
    		else {
    			individual[i] = orig.individual[i];
    		}
    	}
    	
    	n_e = true;
    	fitness = 0.0;
    	n_cutpoints = individual.length;
    }
    
    /**
     * Checks if the current chromosome has been evaluated or not.
     * 
     * @return true, if this chromosome was evaluated previously;
     * false, if it has never been evaluated
     */
    public boolean not_eval() {
        return n_e;
    }
    
    /**
     * Evaluates this chromosome, computing the fitness of the chromosome
     * 
     * @param dataset	Training dataset used in this algorithm
     * @param all_cut_points	Proposed cut points that are selected by the CHC chromosome
     * @param alpha	Coefficient for the number of cut points importance
     * @param beta	Coefficient for the number of inconsistencies importance
     */
    public void evaluate (myDataset dataset, double [][] cut_points, 
    		int initial_cut_points, double alpha, double beta) {
    	
    	int n_selected_cut_points = 0;
    	int [][] discretized_data;
    	boolean [] used;
    	double [] sample;
    	int [] class_distribution;
    	ArrayList <Integer> classes;
    	boolean equal;
    	int incons, max, class_max;
    	
    	// Obtain the number of cut points
    	for (int i=0; i < individual.length; i++) {
    		if (individual[i])
    			n_selected_cut_points++;
    	}
    	
    	// Discretize the dataset according to the selected cut points
    	discretized_data = new int [dataset.getnData()] [dataset.getnInputs()];
    	used = new boolean [dataset.getnData()];
    	classes = new ArrayList <Integer> (dataset.getnData());
    	class_distribution = new int [dataset.getnClasses()];
    	
    	for (int i=0; i<dataset.getnData(); i++) {
    		sample = dataset.getExample(i);
    		for (int j=0; j<dataset.getnInputs(); j++) {
    			discretized_data[i][j] = discretize (j, sample[j], cut_points, dataset);
    		}
    		used[i] = false;
    	}
    	
    	// Use a majority classifier to detect incons
    	/* incons = 0;
    	for (int i=0; i < dataset.getnData(); i++) {
    		if (!used[i]) {
    			classes.clear();
    			
    			classes.add(dataset.getOutputAsInteger(i));
    			used[i] = true;
    			
    			// Search for equal instances
    			for (int j=i+1; j < dataset.getnData(); j++) {
    				if (!used[j]) {
    					equal = true;

        				for (int k=0; k < dataset.getnInputs() && equal; k++) {
        					if (discretized_data[i][k] != discretized_data[j][k]) {
        						equal = false;
        					}
        				}
        				
        				if (equal) {
        					used[j] = true;
        					classes.add(dataset.getOutputAsInteger(j));
        				}
    				}
    			}
    			
    			// Search for incons
    			if (classes.size() > 1) {
    				for (int j=0; j<dataset.getnClasses(); j++) {
    					class_distribution[j] = 0;
    				}
    				for (int j=0; j<classes.size(); j++) {
    					class_distribution[classes.get(j)]++;
    				}
    				
    				max = class_distribution[0];
    				class_max= 0;
    				for (int j=1; j<dataset.getnClasses(); j++) {
    					if (class_distribution[j] > max) {
    						max = class_distribution[j];
    						class_max = j;
    					}
    				}
    				
    				for (int j=0; j<classes.size(); j++) {
    					if (classes.get(j) != class_max)
    						incons++;
    				}
    			}
    		}
    	} */
    	
    	//n_e = false;
    	
    	/* WEKA data set initialization
    	 * Second and Third type of evaluator in precision: WEKA classifier
    	 * */
    	
	    ArrayList<weka.core.Attribute> attributes = new ArrayList<weka.core.Attribute>();
	    double[][] ranges = dataset.getRanges();
	    //weka.core.Instance instances[] = new weka.core.Instance[discretized_data.length];
	    
	    /*Attribute adaptation to WEKA format*/
	    for (int i=0; i< dataset.getnInputs(); i++) {
	    	List<String> att_values = new ArrayList<String>();
    		if(cut_points[i] != null) {
	    		for (int j=0; j < cut_points[i].length + 1; j++)
		    		att_values.add(new String(Integer.toString(j)));
    		} else {
    			for (int j = (int) ranges[i][0]; j <= ranges[i][1]; j++) 
    				att_values.add(new String(Integer.toString(j)));
    		}
    		weka.core.Attribute att = 
	    			new weka.core.Attribute("At" + i, att_values, i);
    	    attributes.add(att);
	    }
	    
	    List<String> att_values = new ArrayList<String>();
	    for (int i=0; i<dataset.getnClasses(); i++) {
	    	att_values.add(new String(Integer.toString(i)));
	    }
	    attributes.add(new weka.core.Attribute("Class", att_values, dataset.getnInputs()));

    	/*WEKA data set construction*/
	    weka.core.Instances datatrain = new weka.core.Instances(
	    		"CHC_evaluation", attributes, 0);
	    datatrain.setClassIndex(attributes.size() - 1);
	    
	    /*Instances adaptation to WEKA format*/
	    int j;
	    double tmp[];
    	for (int i=0; i < discretized_data.length; i++) {
    		tmp = new double[dataset.getnInputs() + 1];    		
    		for (j=0; j < dataset.getnInputs(); j++)
    			tmp[j] = discretized_data[i][j];	    			 		
    		
    		tmp[j] = dataset.getOutputAsInteger(i);
    		Instance inst = new DenseInstance(1.0, tmp);
    		
    		/* Set missing values */
    		for(j=0; j < dataset.getMissing(i).length; j++) {
    			if(dataset.getMissing(i)[j]) 
    				inst.setMissing(j);
    		}
    		
    		datatrain.add(inst);
    	}    	
    	
    	/* Use unpruned C4.5 to detect errors
    	 * Second type of evaluator in precision: C45 classifier
    	 * c45er is the error counter
    	 * */
    	double c45er = 0;
	    J48 baseTree = new J48();		
	    
	    try {
	    	baseTree.buildClassifier(datatrain);	    		
	    } catch (Exception ex) {
	    	ex.printStackTrace();
	    }
	    
	    for (int i=0; i<discretized_data.length; i++) {
	    	try {
	    		if ((int)baseTree.classifyInstance(datatrain.instance(i)) 
	    				!= dataset.getOutputAsInteger(i)) {
	    			c45er++;
	    		}
		    } catch (Exception ex) {
		    	ex.printStackTrace();
		    }
	    }
	    
    	/* Use simple Naive Bayes to detect errors
    	 * Third type of evaluator in precision: Naive Bayes classifier
    	 * nber is the error counter
    	 * */
    	double nber = 0;
	    NaiveBayes baseBayes = new NaiveBayes();		
	    
	    try {
	    	baseBayes.buildClassifier(datatrain);	    		
	    } catch (Exception ex) {
	    	ex.printStackTrace();
	    }
	    
	    for (int i=0; i<discretized_data.length; i++) {
	    	try {
	    		if ((int)baseBayes.classifyInstance(datatrain.instance(i)) 
	    				!= dataset.getOutputAsInteger(i)) {
	    			nber++;
	    		}
		    } catch (Exception ex) {
		    	ex.printStackTrace();
		    }
	    }
	    
	    double p_err = (double) (nber + c45er)  / (double) (dataset.getnData() * 2);
    	double proportion = (double) (dataset.getnData() * 2) / (double) initial_cut_points;
    	double perc_points= (double) n_selected_cut_points / (double) initial_cut_points;
        /* fitness = alpha * ((double) n_selected_cut_points / (double) initial_cut_points) 
        		+ beta * proportion * ((double) incons / (double) dataset.getnData()) ;*/
    	fitness = alpha * perc_points + beta * proportion * p_err;
        n_cutpoints = n_selected_cut_points;
        perc_err = p_err;
        //inconsistencies = incons;
        //System.out.println(fitness);
    }
    
    /**
     * Obtains the fitness associated to this CHC_Chromosome, its fitness measure
     * 
     * @return	the fitness associated to this CHC_Chromosome
     */
    public double getFitness() {
    	return fitness;
    }
    
    /**
     * Obtains the Hamming distance between this and another chromosome
     * 
     * @param ch_b	Other chromosome that we want to compute the Hamming distance to
     * @return	the Hamming distance between this and another chromosome
     */
    public int hammingDistance (CHC_Chromosome ch_b) {
    	int i;
    	int dist = 0;
    	
    	if (individual.length != ch_b.individual.length) {
    		System.err.println("The CHC Chromosomes have different size so we cannot combine them");
    		System.exit(-1);
    	}
    	
    	for (i=0; i<individual.length; i++){
    		if (individual[i] != ch_b.individual[i]) {
    			dist++;
    		}
    	}

    	return dist;
    }
    
    
    /**
     * Obtains a new pair of CHC_chromosome from this chromosome and another chromosome, swapping half the differing bits at random
     * 
     * @param ch_b	Other chromosome that we want to use to create another chromosome
     * @return	a new pair of CHC_chromosome from this chromosome and the given chromosome
     */
    public ArrayList <CHC_Chromosome> createDescendants (CHC_Chromosome ch_b) {
    	int i, pos;
    	int different_values, n_swaps;
    	int [] different_position;
    	CHC_Chromosome descendant1 = new CHC_Chromosome();
    	CHC_Chromosome descendant2 = new CHC_Chromosome();
    	ArrayList <CHC_Chromosome> descendants;
    	
    	if (individual.length != ch_b.individual.length) {
    		System.err.println("The CHC Chromosomes have different size so we cannot combine them");
    		System.exit(-1);
    	}
    	
    	different_position = new int [individual.length];
    	
    	descendant1.individual = new boolean[individual.length];
    	descendant2.individual = new boolean[individual.length];
    	
    	different_values = 0;
    	for (i=0; i<individual.length; i++){
    		descendant1.individual[i] = individual[i];
    		descendant2.individual[i] = ch_b.individual[i];
    		
    		if (individual[i] != ch_b.individual[i]) {
    			different_position[different_values] = i;
    			different_values++;
    		}
    	}
    	
    	n_swaps = different_values/2;
    	
    	if ((different_values > 0) && (n_swaps == 0))
    		n_swaps = 1;
    	
    	for (int j=0; j<n_swaps; j++) {
    		different_values--;
    		pos = Randomize.Randint(0, different_values);
    		
    		boolean tmp = descendant1.individual[different_position[pos]];
    		descendant1.individual[different_position[pos]] = descendant2.individual[different_position[pos]];
    		descendant2.individual[different_position[pos]] = tmp;
    		
    		different_position[pos] = different_position[different_values];
    	}
    	
    	descendant1.n_e = true;
    	descendant2.n_e = true;
    	descendant1.fitness = 0.0;
    	descendant2.fitness = 0.0;
    	descendant1.n_cutpoints = individual.length;
    	descendant2.n_cutpoints = individual.length;
    	
    	descendants = new ArrayList <CHC_Chromosome> (2);
    	descendants.add(descendant1);
    	descendants.add(descendant2);

    	return descendants;
    }    
    
    /**
     * Obtains a new pair of CHC_chromosome from this chromosome and another chromosome, 
     * swapping half the differing bits at random
     * 
     * @param ch_b	Other chromosome that we want to use to create another chromosome
     * @return	a new pair of CHC_chromosome from this chromosome and the given chromosome
     */
    public ArrayList <CHC_Chromosome> createDescendants (CHC_Chromosome ch_b, double prob0to1Rec) {
    	int i;
    	CHC_Chromosome descendant1 = new CHC_Chromosome();
    	CHC_Chromosome descendant2 = new CHC_Chromosome();
    	ArrayList <CHC_Chromosome> descendants;
    	
    	if (individual.length != ch_b.individual.length) {
    		System.err.println("The CHC Chromosomes have different size so we cannot combine them");
    		System.exit(-1);
    	}
    	
    	descendant1.individual = new boolean[individual.length];
    	descendant2.individual = new boolean[individual.length];
    	
    	for (i=0; i<individual.length; i++){
    		descendant1.individual[i] = individual[i];
    		descendant2.individual[i] = ch_b.individual[i];
    		
    		if ((individual[i] != ch_b.individual[i]) && Randomize.Rand() < 0.5) {
    			if (descendant1.individual[i]) 
    				descendant1.individual[i] = false;
				else if (Randomize.Rand() < prob0to1Rec) 
					descendant1.individual[i] = true;
    			
				if (descendant2.individual[i]) 
					descendant2.individual[i] = false;
				else if (Randomize.Rand() < prob0to1Rec) 
					descendant1.individual[i] = true;
			}
    	}
    	
    	descendant1.n_e = true;
    	descendant2.n_e = true;
    	descendant1.fitness = 0.0;
    	descendant2.fitness = 0.0;
    	descendant1.n_cutpoints = individual.length;
    	descendant2.n_cutpoints = individual.length;
    	
    	descendants = new ArrayList <CHC_Chromosome> (2);
    	descendants.add(descendant1);
    	descendants.add(descendant2);

    	return descendants;
    }
    
    /**
     * Obtain the boolean array representing the CHC Chromosome
     * 
     * @return	boolean array selecting rules from a rule base
     */
    public boolean [] getIndividual () {
    	return individual;
    }
    
    
    /**
     * Obtains the discretized value of a real data for an attribute considering the
     * cut points vector given and the individual information 
     * 
     * @param attribute	Position of the attribute that is associated to the given value
     * @param value	Real value we want to discretize according to the considered cutpoints
     * @param cut_points	Proposed cut points that are selected by the CHC chromosome
     * @param dataset	Training dataset used in this algorithm
     * @return	the integer value associated to the discretization done
     */
	private int discretize (int attribute, double value, double [][] cut_points, myDataset dataset) {
		int index_att, index_values, j;
		
		if (dataset.getTipo(attribute) == myDataset.NOMINAL) {
			return ((int)Math.round(value));
		}
		
		if (cut_points[attribute] == null) 
			return 0;
		
		index_att = 0;
		for (int i=0; (i<dataset.getnInputs()) && (i<attribute); i++) {
			if (cut_points[i] != null) {
				index_att += cut_points[i].length;
			}
		}
		
		index_values = 0;
		j = 0;
		for (int i=index_att; i<(index_att+cut_points[attribute].length); i++) {
			if ((value < cut_points[attribute][j]) && (individual[i])) { 
				return index_values;
			}
			
			if (individual[i]) {
				index_values++;
			}
			j++;
		}
		
		return index_values++;
	}
	
	  public String toString() {
		  String output = "";
		  
		  for (int i=0; i<individual.length; i++) {
			  if (individual[i]) {
				  output = output + "1 ";
			  }
			  else {
				  output = output + "0 ";
			  }
		  }
		  
		  return (output);
	  }
    
    /**
     * Compares this object with the specified object for order, according to the fitness measure 
     * 
     * @return a negative integer, zero, or a positive integer as this object is less than, equal to, or greater than the specified object
     */
    public int compareTo (Object aThat) {
        final int BEFORE = -1;
        final int EQUAL = 0;
        final int AFTER = 1;
        
    	if (this == aThat) return EQUAL;
    	
    	final CHC_Chromosome that = (CHC_Chromosome)aThat;
    	
    	if (this.fitness < that.fitness) return BEFORE;
        if (this.fitness > that.fitness) return AFTER;
        
        if (this.n_cutpoints < that.n_cutpoints) return AFTER;
        if (this.n_cutpoints > that.n_cutpoints) return BEFORE;
        return EQUAL;
    }
    
}

