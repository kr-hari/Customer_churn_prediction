import pdb
import csv
import pandas as pd 

def split_target(dset, target_col_name):

	target = dset[target_col_name]
	del dset[target_col_name]
	print("Successfully removed target column")
	target = target.replace({"Yes":1,"No":0})
	# pdb.set_trace()
	return dset,target


def pd_df_from_csv(filename, sampled = False, train = True, colname = "", split_target_ =True, verbose = False):
	""" Function to get a pandas dataframe object from after reading data from csv file
	sampled = True : in case of a smaller dataset for testing code
	Returns :
	dset -> a pandas dataframe object with data"""
	if sampled:
		dset = pd.read_csv(filename, nrows = 100, verbose = verbose)
	else:
		dset = pd.read_csv(filename, verbose = verbose)
	print("Size of dataset (before splitting target) : ", dset.shape)
	if train and split_target_:
		return split_target(dset,colname)	
	elif train:
		return dset
	else:
		client_id = dset["Client ID"]
		del dset["Client ID"]
		return dset, client_id
	
def remove_null(dset, threshold = 0.4, verbose = False):
	""" Function to remove columns with more than threshold% of null values 
	Inputs : 
	dset -> Dataset
	threshold -> threshold value of null values, above which columns will be removed
	Output :
	dset -> Processed Dataset"""

	rows, cols = dset.shape
	ts_null_values = int(threshold * rows)
	dset = dset[dset.columns[dset.isna().sum()<ts_null_values]]
	if verbose:
		print("Removed columns with >",threshold*100,"% null values")
		print("New Dataset size : ", dset.shape)
	return dset

def remove_correlation(dset, threshold = 0.8, verbose = False):
	"""Function to remove columns with more than threshold correlation 
	Inputs : 
	dset -> Dataset
	threshold -> absolute threshold corrleation, above which columns will be removed
	Output :
	dset -> Processed Dataset"""

	col_corr = set() # Set of all the names of deleted columns
	corr_matrix = dset.corr()
	deleted_cols = {}
	for i in range(len(corr_matrix.columns)):
		for j in range(i):
			if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
				colname = corr_matrix.columns[i] # getting the name of column
				deleted_cols[colname] = {'Related column': corr_matrix.columns[j],"Correlation":corr_matrix.iloc[i,j]}
				col_corr.add(colname)
				if colname in dset.columns:
					del dset[colname] # deleting the column from the dataset
	if verbose:
		print("Removed Columns : ")
		for key,value in deleted_cols.items():
			print(key," : ",value)
		print("Removed columns with >",threshold*100,"correlation")
		print("New Dataset size : ", dset.shape)

def rename_fields(dset, dictionary):

	cols = list(dset.columns)
	for i,col  in enumerate(cols):
		for (key,value) in dictionary.items():
			if key in col:
				cols[i] = col.replace(key,value)
				col = cols[i]
	
	for i,column in enumerate(cols):
		cols[i] = " ".join(column.split())

	dset.columns = cols
	return dset

def replace_col_vals(dset,dict_list):
	# pdb	.set_trace()
	for dict_ in dict_list:
		dset = dset.replace(dict_)
	return dset

def make_quarterly_data(dset,columns):

	for col in columns :
		for quarter in range(4):
			dset[col+"_sum_"+str(quarter)] = dset[col+" TP "+str(quarter*3)]
			del dset[col+" TP "+str(quarter*3)]
			for j in range(1,3):
				dset[col+"_sum_"+str(quarter)] += dset[col+" TP "+str(quarter*3+j)]
				del dset[col+" TP "+str(quarter*3+j)]
			dset[col+"_sum_"+str(quarter)]/=3
	return dset