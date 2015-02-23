import java.io.File;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.Loader;

/**
 * The Train class is responsible for loading the training data, instantiating a
 * Classifier, then building the classifier instance with the training data. It
 * then serializes the Classifier to disk for other operations to use.
 * 
 * 
 * @author Hardik Dasadia
 * 
 */
public class C45Trainer {

	public static void generateModels() throws Exception {
		
		for(int i=1;i<=10;i++){
			/*
			 * First we load the training data from our ARFF file
			 */
			ArffLoader trainLoader = new ArffLoader();
			trainLoader.setSource(new File("C:/Users/Hardik/workspace/Classification/Dataset/Training/train"+i+".arff"));
			trainLoader.setRetrieval(Loader.BATCH);
			Instances trainDataSet = trainLoader.getDataSet();

			/*
			 * Now we tell the data set which attribute we want to classify, in our
			 * case, we want to classify the last column A16 which indicates whether customer passed credit check or not
			 */
			Attribute trainAttribute = trainDataSet.attribute(15);
			trainDataSet.setClass(trainAttribute);

			/*
			 * Create a new Classifier of type J48 and configure it.
			 */
			 J48 classifier = new J48();
			 classifier.setDebug(true);

			/*
			 * Now we train the classifier
			 */
			classifier.buildClassifier(trainDataSet);

			/*
			 * We are done training the classifier, so now we serialize it to disk
			 */
			SerializationHelper.write("C:/Users/Hardik/workspace/Classification/Model/J48_"+i+".model", classifier);
			System.out.println("Saved trained model to J48_"+i+".model");
		}
	}
}