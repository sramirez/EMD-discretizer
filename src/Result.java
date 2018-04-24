package keel.Algorithms.Discretizers.ECPSRD;

class Result {
	 
	  public boolean[] best_chromosome;
	  public double[][] selected_cut_points;
	 
	  public Result (boolean[] best_chromosome, double[][] selected_cut_points) {
	    this.best_chromosome = best_chromosome;
	    this.selected_cut_points = selected_cut_points;
	  }
	 
	  public boolean[] getBest() { 
		  return best_chromosome; 
	  }
	  
	  public double[][] getCutPoints() { 
		  return selected_cut_points; 
	  }
}