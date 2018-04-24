package keel.Algorithms.Discretizers.ECPSRD;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import keel.Algorithms.Discretizers.Basic.Discretizer;
import keel.Algorithms.Discretizers.ECPSRD.*;

import org.core.Randomize;

/**
 * <p>Title: ECPSD </p>
 *
 * <p>Description: It contains the implementation of the CHC multivariate discretizer with a historical of cut points.
 * It makes a faster convergence of the algorithm CHC_MV. </p>
 * 
 * <p>Company: KEEL </p>
 *
 * @author Written by Sergio Ramirez (University of Granada) (10/01/2014)
 * @version 1.5
 * @since JDK1.5
 */

public class ECPSRD {
	
	private long seed;
	private double [][] actual_cut_points;
	private myDataset dataset;
	
	private ArrayList <CHC_Chromosome> population;
	
	private int max_cut_points;
	private int n_cut_points;
	
	private int max_eval;
	private int n_eval;
	
	private int pop_length;
	
	private double threshold;
	private double r;
	private double alpha, beta;
	private double best_fitness;

	private double prob1to0Rec;
	private double prob1to0Div;  
	
	private double pEvaluationsForReduction; 
	private double pReduction;

	private int n_restart_not_improving;
	private static int PROPER_SIZE_CHROMOSOME = 1000;
	
    /**
     * Default constructor
     */
    public ECPSRD () { }
    
    /**
     * Creates a CHC object with its parameters
     * 
     * @param pop	Population of rules we want to select
     */
    public ECPSRD (long seed, myDataset current_dataset, double [][] pop, int eval, int popLength, 
    		double restart_per, double alpha_fitness, double beta_fitness, double pr0to1Rec, 
    		double pr0to1Div, double pEvaluationsForReduction, double pReduction) {
    	this.seed = seed;
    	dataset = current_dataset;
    	actual_cut_points = pop;
    	max_eval = eval;
    	pop_length = popLength;
    	r = restart_per;
    	alpha = alpha_fitness;
    	beta = beta_fitness;
    	prob1to0Rec = pr0to1Rec;
    	prob1to0Div = pr0to1Div; 
    	this.pEvaluationsForReduction = pEvaluationsForReduction;
    	this.pReduction = pReduction;
    	
    	max_cut_points = 0;
    	for (int i=0; i<dataset.getnInputs(); i++) {
    		if (pop[i] != null)
    			max_cut_points += pop[i].length;
    	}
    	n_cut_points = max_cut_points;
    	
    	population = new ArrayList <CHC_Chromosome> (pop_length);
    	best_fitness = 100.0;
    }
    
    /**
     * Run the CHC algorithm for the data in this population
     * 
     * @return	boolean array with the rules selected for the final population
     */
    public Result runCHC () {
    	ArrayList <CHC_Chromosome> C_population;
    	ArrayList <CHC_Chromosome> Cr_population;
    	boolean pop_changes;
    	
    	n_eval = 0;
    	int n_reduction = 0;
    	int n_restart = 0;
    	threshold = (double) n_cut_points / 4.0;
    	n_restart_not_improving = 0;
    	int next_reduction = 1;
    	boolean reduction = true;

    	int[] cut_points_log = new int[n_cut_points];
    	
    	//initPopulation(initials_chrom);
    	initPopulation();
    	evalPopulation();    	
    	
    	do {
    		
    		// Reduction?
    		reduction = (n_cut_points * (1 - pReduction) > PROPER_SIZE_CHROMOSOME) &&
    				(n_eval / (max_eval * pEvaluationsForReduction) > next_reduction);
    		if (reduction) {
    			// We reduce the population, it is not evaluated
    			reduction(cut_points_log, ((CHC_Chromosome)population.get(0)).getIndividual());    	    	
    			cut_points_log = new int[n_cut_points];
    			next_reduction++;

    			// Next time we do a restart 
    			// (population do not need be evaluated, best chrom is keeped equal)
    			restartPopulation();
    			//restartPopulation();
    			threshold = Math.round(r * (1.0 - r) * (double) n_cut_points);
			best_fitness = 100.0;
    			evalPopulation();
    			n_reduction++;
    			if(n_cut_points * (1 - pReduction) <= PROPER_SIZE_CHROMOSOME) {
    				System.out.println("No more reductions!");
    			}
    		}
    		
    		// Select for crossover
    		C_population = randomSelection();
    		// Cross selected individuals
    		Cr_population = recombine (C_population);
    		// Evaluate new population
    		evaluate (Cr_population);
    		
    		// Select individuals for new population
    		pop_changes = selectNewPopulation (Cr_population);
    		
    		// Maintain a historical of the most selected cut points after selecting new populations
    		if (reduction) {
	    		for(int i=0; i < n_cut_points; i++) {
	    			if(population.get(0).getIndividual()[i]) 
	    				cut_points_log[i]++;
	    		}
    		}
    		
    		// Check if we have improved or not
    		if (!pop_changes) {
    			threshold--;
    		}
    		
    		// If we do not improve our current population for several trials, then we should restart the population
    		if (threshold < 0) {
    			System.out.println("Restart!!");
    			restartPopulation();
    			threshold = Math.round(r * (1.0 - r) * (double) n_cut_points);
    	    	best_fitness = 100.0;
    			n_restart_not_improving++;
    			evalPopulation();
    			n_restart++;
    		}
    		
    		//System.out.println("CHC procedure: " + n_eval + " of " + max_eval + " evaluations. Best fitness is " + best_fitness);
    		//for (int i=0; i<pop_length; i++) 
    		//	System.out.println(population.get(i));
    	} while ((n_eval < max_eval) && (n_restart_not_improving < 5));

    	// The evaluations have finished now, so we select the individual with best fitness
    	Collections.sort(population);
    	CHC_Chromosome best = population.get(0);
    	System.out.println("Best Chr.: " + best.toString());
    	System.out.println("Best Fitness: "  + best.fitness);
    	System.out.println("Number of selected cutpoints/max points: " 
    			+ best.n_cutpoints + "/" + max_cut_points);
    	System.out.println("Error in train: " + best.perc_err);
    	return new Result(best.getIndividual(), actual_cut_points);
    }
    
    private void reduction(int[] cut_points_log, boolean[] best_chr){

    	ArrayList<RankingPoint> candidatePoints = new ArrayList<RankingPoint>(cut_points_log.length);
    	List<RankingPoint> newPoints = new ArrayList<RankingPoint>(cut_points_log.length);
		int reduced_size = (int) ((1 - pReduction) * cut_points_log.length);
    	
		// Maintain the best chromosome' points and the rest ones are used to form a ranking
    	for(int i = 0; i < cut_points_log.length; i++){
    		RankingPoint point = new RankingPoint(i, cut_points_log[i]);
    		if(best_chr[i])
    			newPoints.add(point);
    		else
    			candidatePoints.add(point);
    	}
		
    	// Select the best ranked points to complete the reduced chromosome
		int rest_size = reduced_size - newPoints.size();
		if(rest_size > 0) { 
			
			if(candidatePoints.size() > rest_size) {
				// Order the points by ranking (descending)
				Collections.sort(candidatePoints, new ComparePointsByRank());
				int pivot = rest_size - 1;
				int last_rank = candidatePoints.get(pivot).getRank();
				
				if (last_rank != candidatePoints.get(pivot + 1).getRank()) {
					// We add all best ranked candidates until completing the reduced chrom
					newPoints.addAll(candidatePoints.subList(0, rest_size));	
				} else {
					// We have to select the last ranked randomly
					int first_pos = 0;
					for(int i = pivot; i >= 0; i--) {
						if(last_rank != candidatePoints.get(i).getRank()){
							first_pos = i + 1;
							break;
						}
					}
					
					int last_pos = candidatePoints.size();
					for(int i = pivot; i < candidatePoints.size(); i++) {
						if(last_rank != candidatePoints.get(i).getRank()){
							last_pos = i;
							break;
						}
					}
					
					List<RankingPoint> lastRanked = candidatePoints.subList(first_pos, last_pos);
					Random r = new Random(seed);
					
					// We remove the last ranked elements until we achive the given size
					while(lastRanked.size() + first_pos > rest_size)
						lastRanked.remove(r.nextInt(lastRanked.size()));
					
					newPoints.addAll(candidatePoints.subList(0, first_pos));
					newPoints.addAll(lastRanked);
				}
			} else {
				// All candidate points fit in the new chrom
				newPoints.addAll(candidatePoints);	
			}
			
		} else {
			System.out.println("Limit of reduction already reached");
		}
		
    	// Order the points by position (id)
    	Collections.sort(newPoints, new ComparePointsByID());
    	
    	// Reduce the actual matrix of cut points using the most selected points' positions
    	int index_points = 0;
    	int index_att = 0;
    	double[][] new_matrix = new double[actual_cut_points.length][];
		for(int i = 0; (i < actual_cut_points.length) && (index_points < newPoints.size()); i++) {
			if(actual_cut_points[i] != null) {
				List<Double> lp = new ArrayList<Double>();
				while(newPoints.get(index_points).id < index_att + actual_cut_points[i].length){
					lp.add(actual_cut_points[i][newPoints.get(index_points).id - index_att]);
					if(++index_points >= newPoints.size()){ break; }
				}

				new_matrix[i] = new double[lp.size()];
				for(int j = 0; j < lp.size(); j++) {
					new_matrix[i][j] = lp.get(j);
				}

				index_att += actual_cut_points[i].length;
			}				
		}
		actual_cut_points = new_matrix;
		
		// Reduce the size of the chromosomes according to the number of points
		n_cut_points = newPoints.size();
		
		for(int i = 0; i < population.size(); i++) {
			boolean[] old_chr = population.get(i).getIndividual();
			boolean[] new_chr = new boolean[n_cut_points];
			for(int j = 0; j < newPoints.size(); j++){
				new_chr[j] = old_chr[newPoints.get(j).id];
			}
			population.set(i, new CHC_Chromosome(new_chr));
		}   	
    	
    	//evalPopulation();
    }
    
    public CHC_Chromosome applyDiscretizer(Discretizer discretizer) { 
    	
    	discretizer.buildCutPoints(dataset.asInstanceSet());
    	double[][] points = discretizer.getCutPoints();
    	boolean[] chr = new boolean[n_cut_points];
    	int index_att = 0;
    
    	for(int i = 0; i < actual_cut_points.length; i++) {
			if(actual_cut_points[i] != null) {
				if(points[i] != null) {
					for(int j = 0, z = 0; z < points[i].length; z++) {
						while(j < actual_cut_points[i].length) {
							if(actual_cut_points[i][j] == points[i][z]){
								chr[index_att + j] = true;
								break;
							} else {
								j++;
							}
						}			
					}
				}
				index_att += actual_cut_points[i].length;
			}
		}
    	
    	return new CHC_Chromosome(chr);
    }
    
    /**
     * Apply a local search algorithm to the resulting population before evaluating it
     * @param old_pop Population crossed
     * 
     * @return	boolean array with the rules selected for the final population
     */
    public ArrayList<CHC_Chromosome> applyLS(Discretizer LSdiscretizer, ArrayList<CHC_Chromosome> old_pop) {
    	
    	Integer[][] selected_cut_points = new Integer[dataset.getnInputs()][];
		ArrayList<CHC_Chromosome> new_population = new ArrayList<CHC_Chromosome>();
		int n_chr = 0;
		
		for(CHC_Chromosome chr: old_pop) {
			// Convert selected cut positions from binary format to integer (creating a matrix)
			int index_att = 0;
			for(int i = 0; i < actual_cut_points.length; i++) {
				List<Integer> positions = new ArrayList<Integer>();
				if(actual_cut_points[i] != null) {
					for(int j = 0; j < actual_cut_points[i].length; j++) {
						if(chr.getIndividual()[index_att + j]) positions.add(j);
					}   				
					selected_cut_points[i] = positions.toArray(new Integer[positions.size()]);
					index_att += actual_cut_points[i].length;
				}
			}
			
			// Apply the discretizer to the pre-selected data (previously initialized)
			Integer[][] newPoints = LSdiscretizer.discretizeAttributes(selected_cut_points);
			
			// Transform again the selected positions to a binary format
			boolean[] new_chr = new boolean[chr.getIndividual().length];
			index_att = 0;
			for(int i = 0; i < actual_cut_points.length; i++) {
				if(actual_cut_points[i] != null) {
					for(int pos: newPoints[i]) {
						new_chr[index_att + pos] = true;
					} 				
					index_att += actual_cut_points[i].length;
				}
			}
			
			new_population.add(new CHC_Chromosome(new_chr));
			n_chr++;
		}

		return new_population;
    }
    
    
    /**
     * Creates several population individuals randomly. The first individual has all its values set to true
     */
    private void initPopulation () {
    	CHC_Chromosome current_chromosome = new CHC_Chromosome (n_cut_points, true);
    	population.add(current_chromosome);
    	
    	for (int i=1; i<pop_length; i++) {
    		current_chromosome = new CHC_Chromosome (n_cut_points);
    		population.add(current_chromosome);
    	}
    }
    
    /**
     * Creates several population individuals randomly. The first individual has all its values set to true
     */
    private void initPopulation (ArrayList<CHC_Chromosome> init_chromosomes) {

    	population.addAll(init_chromosomes);
    	
    	CHC_Chromosome current_chromosome = new CHC_Chromosome (n_cut_points, true);
    	population.add(current_chromosome);
    	
    	for (int i = population.size(); i < pop_length; i++) {
    		current_chromosome = new CHC_Chromosome (n_cut_points);
    		population.add(current_chromosome);
    	}
    }
    
    
    /**
     * Evaluates the population individuals. If a chromosome was previously evaluated we do not evaluate it again
     */
    private void evalPopulation () {
    	double ind_fitness;
    	
        for (int i = 0; i < pop_length; i++) {
            if (population.get(i).not_eval()) {
            	population.get(i).evaluate(dataset, actual_cut_points, max_cut_points, alpha, beta);
            	n_eval++;
            }
        	
        	ind_fitness = population.get(i).getFitness();
        	if (ind_fitness < best_fitness) {
        		best_fitness = ind_fitness;            		
        	}
        }
    }
    
    /**
     * Selects all the members of the current population to a new population ArrayList in random order
     * 
     * @return	the current population in random order
     */
    private ArrayList <CHC_Chromosome> randomSelection() {
    	ArrayList <CHC_Chromosome> C_population;
    	int [] order;
    	int pos, tmp;
    	
    	C_population = new ArrayList <CHC_Chromosome> (pop_length);
    	order = new int[pop_length];
    	
    	for (int i=0; i<pop_length; i++) {
    		order[i] = i;
    	}
    	
    	for (int i=0; i<pop_length; i++) {
    		pos = Randomize.Randint(i, pop_length-1);
    		tmp = order[i];
    		order[i] = order[pos];
    		order[pos] = tmp;
    	}
    	
    	for (int i=0; i<pop_length; i++) {
    		C_population.add(new CHC_Chromosome(((CHC_Chromosome)population.get(order[i]))));
    	}
    	
    	return C_population;
    }
    
    /**
     * Obtains the descendants of the given population by creating the most different descendant from parents which are different enough
     * 
     * @param original_population	Original parents used to create the descendants population
     * @return	Population of descendants of the given population
     */
    private ArrayList <CHC_Chromosome> recombine (ArrayList <CHC_Chromosome> original_population) {
    	ArrayList <CHC_Chromosome> Cr_population;
    	int distHamming, n_descendants;
    	CHC_Chromosome main_parent, second_parent;
    	ArrayList <CHC_Chromosome> descendants;
    	
    	n_descendants = pop_length;
    	if ((n_descendants%2)!=0)
    		n_descendants--;
    	Cr_population = new ArrayList <CHC_Chromosome> (n_descendants);
    	
    	for (int i=0; i<n_descendants; i+=2) {
    		main_parent = (CHC_Chromosome)original_population.get(i);
    		second_parent = (CHC_Chromosome)original_population.get(i+1);
    		
    		distHamming = main_parent.hammingDistance(second_parent);
    		
    		if ((distHamming/2.0) > threshold) {
    			descendants = main_parent.createDescendants(second_parent, prob1to0Rec);
    			//descendants = main_parent.createDescendants(second_parent);
    			Cr_population.add((CHC_Chromosome)descendants.get(0));
    			Cr_population.add((CHC_Chromosome)descendants.get(1));
    		}
    	}
    	
    	return Cr_population;
    }
    
    /**
     * Evaluates the given individuals. If a chromosome was previously evaluated we do not evaluate it again
     * 
     * @param pop	Population of individuals we want to evaluate
     */
    private void evaluate (ArrayList <CHC_Chromosome> pop) {
    	for (int i = 0; i < pop.size(); i++) {
            if (pop.get(i).not_eval()) {
            	pop.get(i).evaluate(dataset, actual_cut_points, max_cut_points, alpha, beta);
            	n_eval++;
            }
        }
    }
    
    /**
     * Replaces the current population with the best individuals of the given population and the current population
     * 
     * @param pop	Population of new individuals we want to introduce in the current population
     * @return true, if any element of the current population is changed with other element of the new population; false, otherwise
     */
    private boolean selectNewPopulation (ArrayList <CHC_Chromosome> pop) {
    	double worst_old_population, best_new_population;
    	
    	// First, we sort the old and the new population
    	Collections.sort(population);
    	Collections.sort(pop);
    	
    	worst_old_population = ((CHC_Chromosome)population.get(population.size()-1)).getFitness();
    	if (pop.size() > 0) {
    		best_new_population = ((CHC_Chromosome)pop.get(0)).getFitness();
    	}
    	else {
    		best_new_population = 0.0;
    	}	
    	
    	//if ((worst_old_population >= best_new_population) || (pop.size() <= 0)) {
    	if ((worst_old_population <= best_new_population) || (pop.size() <= 0)) {
    		return false;
    	} else {
    		ArrayList <CHC_Chromosome> new_pop;
    		CHC_Chromosome current_chromosome;
    		int i = 0;
    		int i_pop = 0;
    		boolean copy_old_population = true;
    		double current_fitness;
    		boolean small_new_pop = false;
    		
    		new_pop = new ArrayList <CHC_Chromosome> (pop_length);
    		
    		// Copy the members of the old population better than the members of the new population
    		do {
    			current_chromosome = (CHC_Chromosome)population.get(i);
    			current_fitness = current_chromosome.getFitness();
    			
    			//if (current_fitness < best_new_population) {
    			if (current_fitness >= best_new_population) {
    				// Check if we have enough members in the new population to create the final population
    				if ((pop_length - i) > pop.size()) {
    					new_pop.add(current_chromosome);
        				i++;
        				small_new_pop = true;
    				} else {
    					copy_old_population = false;
    				}
    			} else {
    				new_pop.add(current_chromosome);
    				i++;
    			}
    		} while ((i < pop_length) && (copy_old_population));
    		
    		while (i < pop_length) {
    			current_chromosome = (CHC_Chromosome)pop.get(i_pop);
    			new_pop.add(current_chromosome);
    			i++;
    			i_pop++;
    		}
    		
    		if (small_new_pop) {
    			Collections.sort(new_pop);
    		}
    		
    		current_fitness = ((CHC_Chromosome)new_pop.get(0)).getFitness();
    		
    		if (best_fitness > current_fitness) {
    			best_fitness = current_fitness;
    			n_restart_not_improving = 0;
    		}
    		
    		population = new_pop;	
        	return true;
    	}
    }
    
    /**
     * Creates a new population using the CHC diverge procedure
     */
    private void restartPopulation () {
    	ArrayList <CHC_Chromosome> new_pop;
    	CHC_Chromosome current_chromosome;
    	
    	new_pop = new ArrayList <CHC_Chromosome> (pop_length);
    	
    	Collections.sort(population);
    	current_chromosome = (CHC_Chromosome)population.get(0);
    	new_pop.add(current_chromosome);
    	
    	for (int i=1; i<pop_length; i++) {
    		//current_chromosome = new CHC_Chromosome (
    		//		(CHC_Chromosome)population.get(0), r);
    		current_chromosome = new CHC_Chromosome (
    				(CHC_Chromosome)population.get(0), r, prob1to0Div);
    		new_pop.add(current_chromosome);
    	}
    	
    	population = new_pop;
    }
    
    class ComparePointsByID implements Comparator<RankingPoint> {
        @Override
        public int compare(RankingPoint o1, RankingPoint o2) {
        	return new Integer(o1.id).compareTo(o2.id);
        }
    }

    // Ascending
    class ComparePointsByRank implements Comparator<RankingPoint> {
        @Override
        public int compare(RankingPoint o1, RankingPoint o2) {
        	return new Integer(o2.rank).compareTo(o1.rank);
        }
    }
}

