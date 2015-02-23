import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ArffLoader;
import weka.core.converters.Loader;

/**
 * The Verify class uses the trained model and the test data to verify the
 * classification actually worked.
 * 
 * @author Hardik Dasadia
 * 
 */
public class RandomVerifier {

	public static void startTesting() throws Exception {

		double totalCorrectPredictions = 0;
		double totalCorrectPercentage = 0;
		
		for(int i=1;i<=10;i++){	//For all 10 folds
			/*
			 * First we load our test data from the .arff file.
			 */
			ArffLoader predictLoader = new ArffLoader();
			predictLoader.setSource(new File("C:/Users/Hardik/workspace/Classification/Dataset/Test/part"+i+".arff"));
			predictLoader.setRetrieval(Loader.BATCH);
			Instances predictDataSet = predictLoader.getDataSet();
			

			/*
			 * Here we set the attribute we want to test the data with
			 */
			Attribute testAttribute = predictDataSet.attribute(15);
			predictDataSet.setClass(testAttribute);

			/*
			 * We still have to remove all string attributes before we can test
			 */
			predictDataSet.deleteStringAttributes();

			/*
			 * Next we load the training data from our ARFF file
			 */
			ArffLoader trainLoader = new ArffLoader();
			trainLoader.setSource(new File("C:/Users/Hardik/workspace/Classification/Dataset/Training/train"+i+".arff"));
			trainLoader.setRetrieval(Loader.BATCH);
			Instances trainDataSet = trainLoader.getDataSet();

			/*
			 * Now we tell the data set which attribute we want to classify
			 */
			Attribute trainAttribute = trainDataSet.attribute(15);
			trainDataSet.setClass(trainAttribute);

			/*
			 * The RandomForest implementation cannot handle columns of type string,
			 * so we remove them for now.
			 */
			trainDataSet.deleteStringAttributes();
			
			/*
			 * Now we read in the serialized model from disk
			 */
			Classifier classifier = (Classifier) SerializationHelper
					.read("C:/Users/Hardik/workspace/Classification/Model/random_forest_"+i+".model");

			/*
			 * Next we will use an Evaluation class to evaluate the performance of
			 * our Classifier.
			 */
			Evaluation evaluation = new Evaluation(trainDataSet);
			evaluation.evaluateModel(classifier, predictDataSet, new Object[] {});

			/*
			 * After we evaluate the Classifier, we write out the summary
			 * information to the screen.
			 */
			System.out.println("---------------------------Fold"+i+"---------------------------");
			System.out.println(classifier);
			System.out.println(evaluation.toSummaryString());
			totalCorrectPredictions = totalCorrectPredictions + (evaluation.correct());
			totalCorrectPercentage = totalCorrectPercentage + (evaluation.pctCorrect());
			
		}
			System.out.println("**************Summary******************");
			System.out.println("Total correct values evaluated by Random Forests 690 instance = "+(double)(totalCorrectPredictions));
			System.out.println("Average correct percentage evaluated by Random Forests = "+(double)(totalCorrectPercentage/10)+"%");
			System.out.println("***************************************");

	}
}