/**
 * <p>Title: Main Class of the Program</p>
 *
 * <p>Description: It reads the configuration file (data-set files and parameters) and launch the CHC multivariate discretizer</p>
 *
 * <p>Company: KEEL</p>
 *
 * @author Written by Victoria Lopez (University of Granada) 03/05/2011
 * @version 1.0
 */

package keel.Algorithms.Discretizers.ECPSRD;

/**
 * <p>Title: Main Class of the Program</p>
 *
 * <p>Description: It reads the configuration file (data-set files and parameters) and launch the CHC multivariate discretizer</p>
 *
 * <p>Company: KEEL</p>
 *
 * @author Written by Victoria Lopez (University of Granada) 03/05/2011
 * @version 1.0
 */
public class Main {

	private parseParameters parameters;
	
    /** Default Constructor */
    public Main() {
    }    

    /**
     * It launches the CHC multivariate discretizer
     * 
     * @param confFile String it is the filename of the configuration file.
     */
    private void execute(String confFile) {
        parameters = new parseParameters();
        parameters.parseConfigurationFile(confFile);
        ECPSRD_Discretizer method = new ECPSRD_Discretizer(parameters);
        method.execute();
    }    
    
    /**
     * Main Program
     * @param args It contains the name of the configuration file<br/>
     * Format:<br/>
     * <em>algorithm = &lt;algorithm name></em><br/>
     * <em>inputData = "&lt;training file&gt;" "&lt;validation file&gt;" "&lt;test file&gt;"</em> ...<br/>
     * <em>outputData = "&lt;training file&gt;" "&lt;test file&gt;"</em> ...<br/>
     * <br/>
     * <em>seed = value</em> (if used)<br/>
     * <em>&lt;Parameter1&gt; = &lt;value1&gt;</em><br/>
     * <em>&lt;Parameter2&gt; = &lt;value2&gt;</em> ... <br/>
     */
    public static void main(String args[]) {
        long t_ini = System.currentTimeMillis();    	
        Main program = new Main();
        System.out.println("Executing Algorithm.");
        program.execute(args[0]);
        long t_fin = System.currentTimeMillis();
        long t_exec = t_fin - t_ini;
        long hours = t_exec / 3600000;
        long rest = t_exec % 3600000;
        long minutes = rest / 60000;
        rest %= 60000;
        long seconds = rest / 1000;
        rest %= 1000;
        System.out.println("Execution Time: " + hours + ":" + minutes + ":" +
                           seconds + "." + rest);        
    }

}
