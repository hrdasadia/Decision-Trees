import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
 
import java.io.File;
 
public class ArffConverter {
  /**
   * takes 2 arguments:
   * - CSV input file
   * - ARFF output file
   */
  public static void convert(String input, String output) throws Exception {
     
    // load CSV
    CSVLoader loader = new CSVLoader();
    loader.setSource(new File(input));
    Instances data = loader.getDataSet();
 
    // save ARFF
    ArffSaver saver = new ArffSaver();
    saver.setInstances(data);
    File outputFile = new File(output);
    saver.setFile(outputFile);
    saver.writeBatch();
  }
}