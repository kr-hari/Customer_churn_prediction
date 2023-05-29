import argparse
import pdb
import os 

import numpy as np
import pandas
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import GridSearchCV

from utils import *
from config import *

import warnings
warnings.filterwarnings("ignore")


def measure(y_actual,y_hat):
	""" Function to find sensitivity, specificity & Accuracy 
	"""
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	y_actual = list(y_actual)
	for i in range(len(y_hat)): 
		if y_actual[i] ==1 and  y_hat[i]==1:
			TP += 1
		if y_hat[i] == 1 and y_actual[i] == 0:
			FP += 1
		if y_hat[i] ==0 and  y_actual[i] == 0:
			TN +=1
		if y_hat[i] == 0 and y_actual[i] == 1:
			FN +=1

	sensitivity = TP/(TP+FN)
	specificity = TN/(TN+FP)
	Accuracy = (TP+TN)/(TP+TN+FP+FN)
	return sensitivity,specificity,Accuracy

def predict(train_dset, test_dset, train_target, model_name, args, client_id):
	"""Function to predict the result in test datasets using trained model"""

	print("Final shape of dataset : ",train_dset.shape)

	# Choose a model
	if model_name == "svr":
		from sklearn.svm import LinearSVC
		method = LinearSVC(max_iter = 10000, tol=1e-6)
	elif model_name == "svc":
		from sklearn.svm import SVC
		method = SVC( tol= 1e-6)
	elif model_name == "grad_boost":
		from sklearn.ensemble import GradientBoostingClassifier as GBC
		method = GBC(learning_rate = 0.05, n_estimators=30000, max_depth = 3, tol =1e-4, subsample= 1,verbose = 1, warm_start=False, random_state=5)
	elif model_name == "voting":
		from sklearn.ensemble import GradientBoostingClassifier as GBC
		from sklearn.ensemble import VotingClassifier as VC
		from sklearn.ensemble import RandomForestClassifier as RFC
		clf1 = RFC(n_estimators=10000,max_depth =8,criterion ='gini',n_jobs =1, random_state=5,oob_score =True)
		clf2 = RFC(n_estimators=5000,max_depth =12,criterion ='gini',n_jobs =1, random_state=5,oob_score =True)
		clf3 = RFC(n_estimators=15000,max_depth =6,criterion ='gini',n_jobs =1, random_state=5,oob_score =True)
		clf4 = RFC(n_estimators=3000,max_depth =16,criterion ='gini',n_jobs =1, random_state=5,oob_score =True)
		clf5 = GBC(learning_rate = 0.07, n_estimators=10000, max_depth = 8, tol =1e-4, subsample= 1,verbose=1)
		clf6 = RFC(n_estimators=10000,max_depth =8,criterion ='entropy',n_jobs =1, random_state=5,oob_score =True)
		method = VC(estimators=[('gbc_0', clf1),('gbc_1', clf2),('gbc_2', clf3),('gbc_3', clf4),('gbc_4', clf5),('gbc_5', clf6)],voting='soft',weights=[1,1,1,1,2,1],n_jobs=6)

	# Predict on the dataset
	print("Method used", method)
	method.fit(train_dset,train_target)
	targets = method.predict(test_dset)

	os.makedirs("Results", exist_ok = True)
	result = pd.DataFrame({"Client ID":client_id,"Client Retention Flag":targets})
	result = result.replace({1:"Yes",0:"No"}) 
	result.to_csv('Results/' + args.save_fname +'.csv',index = None, header = True)
	print("Finished Predicting and Creating file")

def process_dset(dset, args, train = True):
	"""Function to process features. All data processing is done here.
	   Takes in a raw dataset and outputs processed dataset."""
	
	if train:
		# Remove columns with more than threshold% of null values
		dset = remove_null(dset, threshold = 0.4, verbose = False)

	# Rename column names
	dset = rename_fields(dset, dictionary = FIELD_DICTIONARY)

	# Replace column containing texts with numbers
	dict_list = [STARTING_MONTH_DICT,SUBSCRIPTION_DICT,INQUIRY_DICT,ONBOARDED_DICT,INDUSTRY_DICT]
	dset = replace_col_vals(dset, dict_list)

	# Remove unncecessary columns
	try:
		# Inside try block since "Client ID" field will be already removed in case of test dataset
		del dset["Client ID"]
	except:
		pass
	del dset["Company ID"]

	return dset

def train(dset, target, args, model ): 
	"""Function to train dataset using cross validation. 
	Algorihtm can be chosen from a given set"""
	percentage_of_one_class = []
	scores = []
	f1_scores = []
	sensitivities = []
	specificities = []

	# Try different models
	if model == "svr":
		from sklearn.svm import LinearSVC
		method = LinearSVC(max_iter = 10000, tol=1e-6)
	elif model == "svc":
		from sklearn.svm import SVC
		method = SVC( tol= 1e-6)
	elif model == "grad_boost":
		from sklearn.ensemble import GradientBoostingClassifier as GBC
		# method = GBC(learning_rate = 0.04, n_estimators=10, max_depth = 12, subsample = 1,tol =1e-4, verbose = 0)		
		method = GBC(learning_rate = 0.04, n_estimators=15000, max_depth = 3, subsample = 1,tol =1e-4, max_features =None, min_samples_leaf=1,min_samples_split =2,verbose = 1,random_state=None)
	elif model == "random_forest":
		from sklearn.ensemble import RandomForestClassifier
		method = RandomForestClassifier(n_estimators=10000,max_depth =8,n_jobs =-1)
	elif model == "adaboost_dc":
		from sklearn.ensemble import AdaBoostClassifier
		method = AdaBoostClassifier(n_estimators=10000,learning_rate=0.5)
	elif model == "bagging_dc":
		print("Using Bagging DC")
		from sklearn.ensemble import BaggingClassifier
		method = BaggingClassifier(n_estimators = 10000,n_jobs=-1)
	elif model == "voting":
		print("Using Voting Classifier")
		from sklearn.linear_model import LogisticRegression as LR 
		from sklearn.naive_bayes import GaussianNB as GNB 
		from sklearn.ensemble import RandomForestClassifier as RFC 
		from sklearn.ensemble import VotingClassifier as VC 
		from sklearn.ensemble import AdaBoostClassifier as ABC
		from sklearn.ensemble import BaggingClassifier as BC 
		from sklearn.ensemble import GradientBoostingClassifier as GBC
		from sklearn.tree import DecisionTreeClassifier as DTC 
		from sklearn.neighbors import KNeighborsClassifier as KNN
		from sklearn.svm import SVC
		from sklearn.model_selection import GridSearchCV

		clf1 = DTC(max_depth=8)
		clf2 = KNN(n_neighbors=7)
		clf3 = SVC( tol= 1e-5,probability=True)
		clf4 = LR(solver='lbfgs', multi_class='multinomial',random_state=1,max_iter=10000)
		clf5 = GNB()
		clf6 = RFC(n_estimators=100,max_depth =8)
		clf7 = ABC(n_estimators=100,learning_rate=0.1)
		clf8 = BC(n_estimators = 100)
		clf9 = GBC(learning_rate = 0.05, n_estimators=1000, max_depth = 8, tol =1e-4, subsample= 1,verbose=0)
		clf10 = GBC(learning_rate = 0.05, n_estimators=500, max_depth = 12, tol =1e-4, subsample= 1,verbose=0)
		clf11 = GBC(learning_rate = 0.05, n_estimators=1500, max_depth = 6, tol =1e-4, subsample= 1,verbose=1)
		method = VC(estimators=[('gbc_0', clf9),('gbc_1', clf10),('gbc_2', clf11)],voting='soft', weights=[1,1,1],n_jobs=3)
	
	# Print the selected model
	print("Method: ",method)
	# Do k-fold cross validation
	cv = KFold(n_splits = args.k, shuffle = True, random_state = 1)
	for i,(train_index, test_index) in enumerate(cv.split(dset)):
		X_train, X_test, y_train, y_test = dset.iloc[train_index,:], dset.iloc[test_index,:], target.iloc[train_index], target.iloc[test_index]
		method.fit(X_train,y_train)

		# To check whether it is predicting only one class
		prediction = method.predict(X_test)
		_, freqs = np.unique(prediction, return_counts=True)
		# To check whether only one class is predicted
		percentage_of_one_class.append( float(freqs[0])/float((freqs[0]+freqs[1])))		
		score = method.score(X_test,y_test)
		f1_scores.append(f1_score(y_test,prediction))
		sensitivity,specificity,accuracy = measure(y_test,prediction)
		sensitivities.append(sensitivity)
		specificities.append(specificity)
		print("Iteration : ",i+1," - Validation score = ",score," - F1 score",f1_score(y_test,prediction)," - Sensitivity",sensitivity," - Specificity",specificity)
		scores.append(score)
	print("** Statistics of ",model," **")
	print("MEAN : ", np.mean(np.array(scores)),"\t - \t STANDARD DEVIATION : ", np.std(np.array(scores)))
	print("F1 - score")
	print("Mean : ",  np.mean(np.array(f1_scores)),"\t - \t STANDARD DEVIATION : ", np.std(np.array(f1_scores)))
	print("PERCENTAGE OF ANY ONE CLASS : ", percentage_of_one_class)
	print("Specificity")
	print("Mean : ",  np.mean(np.array(specificities)),"\t - \t STANDARD DEVIATION : ", np.std(np.array(specificities)))
	print("Sensitivity")
	print("Mean : ",  np.mean(np.array(sensitivities)),"\t - \t STANDARD DEVIATION : ", np.std(np.array(sensitivities)))
	print("\n\n\n")

def feature_engineer(dset, train = True):
	"""Function to extract new features from existing ones"""
	
	range_of_months = 12
	months = [i for i in range(12)]
	testimonials = ["Testimonials TP "+str(month)for month in months]
	test_array = np.zeros((dset.shape[0],1))

	# Create a new column to see whether client has provided testimonial or not
	for t in testimonials:
		test_array+= np.array(dset[t]).reshape(-1,1)
	dset['Provided_testimonial'] = np.array([test_array>0]).astype('int64').reshape(-1,1)

	# Delete all testimonial columns
	for i in range(range_of_months):
		del dset["Testimonials TP "+str(i)]

	columns = ["No_documents","Social_media_views","No_inquiry","Calls_by_Gartner_sd","No_meetings","No_symposiums","Conferences"]#,"Testimonials"]
	# Create features which has the difference in activities of each month
	for col in columns:
		for i in range(range_of_months-1):
			# dset[col+"_diff_"+str(i)] = dset[col+"_sum_"+str(i+1)] - dset[col+"_sum_"+str(i)]
			dset[col+"_diff_"+str(i)] = dset[col+" TP "+str(i+1)] - dset[col+" TP "+str(i)]
	
	# Engagement in a month
	for month in range(12):
		dset["Engagement_in_month_"+str(month)] = np.zeros((dset.shape[0],1))
		for col in columns:        
			col_val = dset[col+" TP "+str(month)].astype('int64')
			# Set maximum cap
			quantile = 1 if col_val.quantile(0.95)==0 else col_val.quantile(0.95)
			col_val = np.array([quantile if val>quantile  else val for val in col_val ])
			col_val = col_val/col_val.max()
			dset["Engagement_in_month_"+str(month)] += col_val

	# Engagement per activity
	for col in columns:
		dset["Engagement_in_activity_"+str(col)] = np.zeros((dset.shape[0],1))
		for month in range(12):
			col_val = dset[col+" TP "+str(month)].astype('int64')
			quantile = 1 if col_val.quantile(0.95)==0 else col_val.quantile(0.95)
			col_val = np.array([quantile if val>quantile  else val for val in col_val ])
			col_val = col_val/col_val.max()
			dset["Engagement_in_activity_"+str(col)] += col_val
	
	# Add abolute month fields
	dict_abs_month = {}
	for index, row in dset.iterrows():
		month = row["Client Contract Starting Month"]
		for i in range(12):
			try:
				dict_abs_month[(month+i)%12].append(row["Engagement_in_month_"+str(i)])
			except:
				dict_abs_month[(month+i)%12] = []
				dict_abs_month[(month+i)%12].append(row["Engagement_in_month_"+str(i)])
	for key,value in dict_abs_month.items():
		dset["Engagement_in_abs_month_"+str(key)] = np.array(value)
	
	return dset

def feature_selection(dset,target, k_features_highest_score = 0 ):
	"""Function to select the k best features from the entire
	feature space."""
	# To do feature selection
	if k_features_highest_score == 0:
		# Select features using chi-squared test
		from sklearn.feature_selection import SelectKBest
		from sklearn.feature_selection import chi2
		from sklearn.model_selection import cross_val_score

		## Get score using original model
		from sklearn.ensemble import GradientBoostingClassifier as GBC
		method = GBC(learning_rate = 0.05, n_estimators=500, max_depth = 3, tol =1e-4, subsample=1)
		def scorer(estimator,X,y):
			prediction = estimator.predict(X)		
			return f1_score(y,prediction)

		highest_score = 0
		print("\n Doing Feature Selection")
		## Get score using models with feature selection
		for i in range(1, dset.shape[1]+1, 1):
			# Select i features
			select = SelectKBest(score_func=chi2, k=i)
			select.fit(dset, target)
			X_train_poly_selected = select.transform(dset)

			# Model with i features selected
			method.fit(X_train_poly_selected, target)
			scores = cross_val_score(method, X_train_poly_selected, target, scoring=scorer,cv=5)
			print('CV accuracy (number of features = %i): %.3f +/- %.3f' % (i,np.mean(scores), np.std(scores)))
			
			# Save results if best score
			if np.mean(scores) > highest_score:
				highest_score = np.mean(scores)
				std = np.std(scores)
				k_features_highest_score = i
			elif np.mean(scores) == highest_score:
				if np.std(scores) < std:
					highest_score = np.mean(scores)
					std = np.std(scores)
					k_features_highest_score = i
		print("No: of features with best score : ",k_features_highest_score,"\t Score : ",highest_score)

	else:
		# return k best features
		from sklearn.feature_selection import SelectKBest
		from sklearn.feature_selection import chi2

		select = SelectKBest(score_func=chi2, k= k_features_highest_score)
		select.fit(dset, target)
		return pd.DataFrame(select.transform(dset)), select.get_support(indices=True)

def hyper_param_tuning(dset, target, args=None):
	"""Find the best hyper parameter for the GBC model"""
	param_test1 = {'n_estimators': [10000,15000], 'max_depth' : [3,6,12],'min_samples_leaf':[2,4],'min_samples_split' : [2,4]}
	gsearch1 = GridSearchCV(estimator = GBC(learning_rate=0.04,# n_estimators = 10000,# min_samples_split=500,min_samples_leaf=50,max_features='sqrt',max_depth=8
		subsample=1,random_state=10), param_grid = param_test1, scoring='roc_auc',n_jobs=5,iid=False, cv=5,verbose =2)
	print("\n\n**GRID SEARCH**")
	gsearch1.fit(dset,target)
	print("Gradient Scores : ", gsearch1.cv_results_)
	print("Gradient Best Params : ",gsearch1.best_params_)
	print("Best Score : ",gsearch1.best_score_,"\n\n")


def main():
	parser = argparse.ArgumentParser()
	# parser.add_argument("--b_size", help = "Batch size to be used for training", type = int)
	parser.add_argument("--sampled", help = "Use sampled data?", type = int)
	parser.add_argument("--threshold", help = "Threshold correlation for eliminating columns ", type = float , default = 80)
	parser.add_argument("--save_fname", help = "Name in which predction has to be saved", type = str, default="Testing")
	parser.add_argument("--mode", help = "Mode to run - CV or prediction", type = str, default="cv")
	parser.add_argument("--fe", help = "Do feature extraction, default-NO", type = int, default=0)
	parser.add_argument("--fs", help = "Do feature Selection, default-NO", type = int, default=0)
	parser.add_argument("--hp", help = "Hyper Parameter Tuning, default-NO", type = int, default=0)
	parser.add_argument("--k", help = "K for K Fold cross validation", type = int, default=10)
	parser.add_argument("--model", help = "model name for training", type = str, default="grad_boost")
	args = parser.parse_args()

	filename = "Dataset/Train.csv"

	# Get a pandas dataframe object from csv file
	dset, target = pd_df_from_csv(filename, sampled = False, train = True, colname = "Client Retention Flag")
	
	# Check the data
	print(dset.head())

	# Process / Clean-up the dataset
	dset = process_dset(dset = dset, args = args, train = True )
	
	if args.fe ==1:
		# Do feature Engineering
		dset = feature_engineer(dset = dset, train = True)
	if args.fs ==1 :
		# Do feature selection
		dset,col_ids = feature_selection(dset,target,k_features_highest_score=130)
	if args.hp ==1:
		# Find best hyperparameters 
		hyper_param_tuning(dset,target)
		exit()
	if args.mode == "test":
		# Do the same with test dataset
		test_filename = "Dataset/Test.csv"
		test_dset, client_id = pd_df_from_csv(test_filename, sampled = False, train = False, colname = "Client Retention Flag")
		test_dset = process_dset(dset = test_dset, args = args, train = False )
		if args.fe ==1:
			test_dset = feature_engineer(dset = test_dset, train = False)
		if args.fs:
			test_dset = test_dset[[test_dset.columns[x] for x in col_ids]]

	# Train / Predict
	if args.mode == "cv":
		# cv_models = ["voting",bagging_dc",adaboost_dc",random_forest","grad_boost","svc","svr",]
		model = train(dset = dset, target = target, model = args.model, args = args)
	elif args.mode == "test":
		predict(train_dset = dset, test_dset = test_dset, train_target = target, model_name = args.model, args = args, client_id = client_id)
	
main()


# Current Best submission with differences - and grad boost F1-score = 87..
# Gradient boost ( GBC(learning_rate = 0.05, n_estimators=10000, max_depth = 8/12, tol =1e-5, verbose = 1))
# GBC(learning_rate = 0.04, n_estimators=10000, max_depth = 8, tol =1e-4, subsample= 1,verbose = 1, warm_start=True) using feature extraction but no feature sampling - 87.31

# Sensitivity = TP/TP+FN
# Specificity  = TN/TN+FP