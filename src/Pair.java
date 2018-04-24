package keel.Algorithms.Discretizers.ECPSRD;

public class Pair implements Comparable {
	 
	  public int clas;
	  public double proportion;
	 
	  public Pair (int clas, double proportion) {
	    this.clas = clas;
	    this.proportion = proportion;
	  }
	 
	  public int getClas() { 
		  return clas; 
	  }
	  
	  public double getProportion () { 
		  return proportion; 
	  }
	 
	  public void setClas (int newClas) { 
		  clas = newClas;  
	  }
	  
	  public void setProportion (double newProportion) { 
		  proportion = newProportion; 
	  }
	 
	  public String toString() {
		  return ("Class " + clas + " has " + proportion + " of elements");
	  }
	 
	  public boolean equals (Object o) {
		  if (o == null)
			  return false;
		  if (o == this)
			  return true;
		  if (!(o instanceof Pair))
			  return false;
		  
		  Pair p = (Pair) o;
		  
		  if (clas != p.clas)
			  return false;
		  
		  if (proportion != p.proportion)
			  return false;
		  
		  return true;
	  }
	  
	  public int hashCode() {
		  int result = 17;
		  
		  result = 31 * result + clas;
		  result = 31 * result + (int) (Double.doubleToLongBits(proportion)^((Double.doubleToLongBits(proportion) >>> 32)));
		  
		  return result;
	  }
	  
	  public int compareTo (Object o) {
		  Pair otherPair = (Pair) o;
		  
		  if (proportion < otherPair.proportion)
			  return -1;
		  if (proportion > otherPair.proportion)
			  return 1;
		  
		  if (clas < otherPair.clas)
			  return -1;
		  if (clas > otherPair.clas)
			  return 1;	
		  
		  return 0;
	  }
}
