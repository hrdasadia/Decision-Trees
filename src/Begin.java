
public class Begin {

	public static void main(String[] args) throws Exception {

		//STEP 0: Divide original dataset into 10 parts(1/10) to be used for testing 10 fold cross validation
		//and training data for every test, remaining (9/10)
		
		//STEP 1: Convert .csv files to .arff
		for(int i=1;i<=10;i++){
			ArffConverter.convert("C:/Users/Hardik/workspace/Classification/Dataset/Test/part"+i+".csv","C:/Users/Hardik/workspace/Classification/Dataset/Test/part"+i+".arff");
			ArffConverter.convert("C:/Users/Hardik/workspace/Classification/Dataset/Training/train"+i+".csv","C:/Users/Hardik/workspace/Classification/Dataset/Training/train"+i+".arff");
		}
		
		//STEP 2:Generate training models for Random Forests and C4.5 and serialize them to use in next step
		RandomForestTrainer.generateModels();
		C45Trainer.generateModels();
		
		//STEP 3: Verify test data and calculate accuracy using the generated models
		RandomVerifier.startTesting();
		C45Verifier.startTesting();
		
	}

}
