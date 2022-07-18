
import warnings
warnings.filterwarnings("ignore")
from fastsklearnfeature.declarative_automl.fair_data.arff2pandas import load


data_ids = [
					31,  # credit-g => personal status, foreign_worker
					1590,  # adult => sex, race
					1461,  # bank-marketing => age

					42193,#compas-two-years => sex, age, race
					1480,#ilpd => sex => V2
					804, #hutsof99_logis => age,gender
					42178,#telco-customer-churn => gender
					981, #kdd_internet_usage => gender
					40536, #SpeedDating => race
					40945, #Titanic => Sex
					451, #Irish => Sex
					945, #kidney => sex
					446, #prnn_crabs => sex
					1017, #arrhythmia => sex
					957, #braziltourism => sex
					41430, #DiabeticMellitus => sex
					1240, #AirlinesCodrnaAdult sex
					1018, #ipums_la_99-small
					55, #hepatitis
					802,#pbcseq
					38,#sick
					40713, #dis
					1003,#primary-tumor
					934, #socmob
					]

'''
for i in range(len(data_ids)):
	dataset = openml.datasets.get_dataset(data_ids[i], download_data=False)
	print(dataset.name + ': ' )
'''

map_dataset = {}

map_dataset['31'] = 'foreign_worker@{yes,no}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['1590'] = 'sex@{Female,Male}'
map_dataset['1461'] = 'AGE@{True,False}'
map_dataset['42193'] ='race_Caucasian@{0,1}'
map_dataset['1480'] = 'V2@{Female,Male}'
map_dataset['804'] = 'Gender@{0,1}'
map_dataset['42178'] = 'gender@STRING'
map_dataset['981'] = 'Gender@{Female,Male}'
map_dataset['40536'] = 'samerace@{0,1}'
map_dataset['40945'] = 'sex@{female,male}'
map_dataset['451'] = 'Sex@{female,male}'
map_dataset['945'] = 'sex@{female,male}'
map_dataset['446'] = 'sex@{Female,Male}'
map_dataset['1017'] = 'sex@{0,1}'
map_dataset['957'] = 'Sex@{0,1,4}'
map_dataset['41430'] = 'SEX@{True,False}'
map_dataset['1240'] = 'sex@{Female,Male}'
map_dataset['1018'] = 'sex@{Female,Male}'
map_dataset['55'] = 'SEX@{male,female}'
map_dataset['802'] = 'sex@{female,male}'
map_dataset['38'] = 'sex@{F,M}'
map_dataset['40713'] = 'SEX@{True,False}'
map_dataset['1003'] = 'sex@{male,female}'
map_dataset['934'] = 'race@{black,white}'


number_instances = []
number_attributes = []
number_features = []


def get_class_attribute_name(df):
	for i in range(len(df.columns)):
		if str(df.columns[i]).startswith('class@'):
			return str(df.columns[i])

def get_sensitive_attribute_id(df, sensitive_attribute_name):
	for i in range(len(df.columns)):
		if str(df.columns[i]) == sensitive_attribute_name:
			return i

with open(Config.get('data_path') + "/downloaded_arff/" + "42132.arff") as f:
	df = load(f)
	print(df.columns)

	print(df.head())


	#df['TotalCharges@REAL'] = pd.to_numeric(df['TotalCharges@STRING'], errors='coerce')
	df = df.drop(columns=['geolocation@STRING'])
	df = df.drop(columns=['seqid@STRING'])
	df = df.drop(columns=['date_of_stop@STRING'])
	df = df.drop(columns=['time_of_stop@STRING'])
	df = df.drop(columns=['description@STRING'])
	df = df.drop(columns=['location@STRING'])

	df.rename(columns={'race@{ASIAN,BLACK,HISPANIC,NATIVE AMERICAN,OTHER,WHITE}': 'race@{BLACKLIVESMATTER,OTHER}'}, inplace=True)
	df.rename(columns={'violation_type@{Citation,ESERO,SERO,Warning}': 'class@{Other,Warning}'},
			  inplace=True)

	def maprace(v):
		if v == 'BLACK':
			return 'BLACKLIVESMATTER'
		else:
			return 'OTHER'

	def mapviolation(v):
		if v == 'Warning':
			return 'Warning'
		else:
			return 'Other'

	df['race@{BLACKLIVESMATTER,OTHER}'] = df['race@{BLACKLIVESMATTER,OTHER}'].apply(maprace)
	df['class@{Other,Warning}'] = df['class@{Other,Warning}'].apply(mapviolation)



	for i in range(len(df.columns)):
		print(df.columns[i] + ": " + str(len(df[df.columns[i]].unique())))

	with open(Config.get('data_path') + "/downloaded_arff/" + '42132_new.arff', 'w') as ff:
		a2p.dump(df, ff)