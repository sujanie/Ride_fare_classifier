# Load libraries
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import calendar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import array
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score,make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

def cleanData(data, dropcols):

    day_dict = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}

    data['pickup_time'] = pd.to_datetime(data['pickup_time'])
    data['pickup_day'] = data['pickup_time'].apply(lambda x: x.day)
    data['pickup_hour'] = data['pickup_time'].apply(lambda x: x.hour)
    data['pickup_day_of_week'] = data['pickup_time'].apply(lambda x: calendar.day_name[x.weekday()])
    data['pickup_month'] = data['pickup_time'].apply(lambda x: x.month)
    data['pickup_year'] = data['pickup_time'].apply(lambda x: x.year)
    data['pickup_day_of_week'] = data['pickup_day_of_week'].apply(lambda x: day_dict[x])
    data['drop_time'] = pd.to_datetime(data['drop_time'])
    data['drop_day'] = data['drop_time'].apply(lambda x: x.day)
    data['drop_hour'] = data['drop_time'].apply(lambda x: x.hour)
    data['drop_day_of_week'] = data['drop_time'].apply(lambda x: calendar.day_name[x.weekday()])
    data['drop_month'] = data['drop_time'].apply(lambda x: x.month)
    data['drop_year'] = data['drop_time'].apply(lambda x: x.year)
    data['period_of_day']= data['pickup_hour'].map(map_hours)
    data['drop_day_of_week'] = data['drop_day_of_week'].apply(lambda x: day_dict[x])
    data['manhattan_dist'] = manhattan_distance(data['pick_lat'].values,data['pick_lon'].values,data['drop_lat'].values,data['drop_lon'].values)
    #data['center_latitude'] = (data['pick_lat'].values + data['drop_lat'].values) / 2
    #data['center_longitude'] = (data['pick_lon'].values + data['drop_lon'].values) / 2
    data['total_time']=(data['drop_time']-data['pickup_time']).astype('timedelta64[s]')
    data['duration'] = data['duration'].fillna(value = data['total_time'])
    data['additional_fare']=data['additional_fare'].fillna(value=round(data['additional_fare'].mode()[0]))
    data['bearing_dist'] = bearing_array(data['pick_lat'].values, data['pick_lon'].values,
                                                data['drop_lat'].values, data['drop_lon'].values)
    data['haversine_dist'] = haversine_array(data['pick_lat'].values, data['pick_lon'].values,
                                         data['drop_lat'].values, data['drop_lon'].values)

    #data['add_fare']=data['meter_waiting_till_pickup'/data['additional_fare']
    #data['diff']=np.abs(data['total_time']-data['duration'])]
    print(data['total_time'])
    data.drop(dropcols, axis=1, inplace=True)
    imp = IterativeImputer(RandomForestRegressor(n_estimators=5), max_iter=5, random_state=1)
    to_train = ['meter_waiting', 'meter_waiting_fare', 'meter_waiting_till_pickup','fare']
    # perform filling
    for k in to_train:
        mode_df = round(data[k].mode()[0])
        mode_df = round(data[k].mean())
        data[k]=data[k].fillna(mode_df)
    #data[to_train] = pd.DataFrame(imp.fit_transform(data[to_train]), columns=to_train)
    data['manhattan_dist'] = np.log1p(data['manhattan_dist'])
    data['meter_waiting_fare'] = np.log1p(data['meter_waiting_fare'])
    #sns.distplot(data['meter_waiting_fare'])
    #plt.title("meter_waiting_fare")
   # plt.savefig("hist_trans3.png")
    #sns.distplot(data['manhattan_dist'])
    #plt.title("manhattan_dist")
   # plt.savefig("hist_trans4.png")
    #plt.show()
    data=poly_transform(data)
    return data

def poly_transform(data):
    poly = PolynomialFeatures(interaction_only=True)
    sc = RobustScaler()
    to_check=['meter_waiting_till_pickup','additional_fare']
    to_crosscheck = ['meter_waiting', 'meter_waiting_fare']
    feats=['additional_fare','meter_waiting_fare','meter_waiting','meter_waiting_till_pickup']
    crossed_feats = poly.fit_transform(data[to_crosscheck].values)
    crossed_feats = pd.DataFrame(crossed_feats)
    data = pd.concat([data, crossed_feats], axis=1)
    crossed_feats1 = poly.fit_transform(data[to_check].values)
    crossed_feats1 = pd.DataFrame(crossed_feats1)
    data = pd.concat([data, crossed_feats1], axis=1)
    sc_data = sc.fit_transform(data[feats])
    sc_data=pd.DataFrame(sc_data)
    data = pd.concat([data, sc_data], axis=1)
    return data
def map_hours(x):
    if x in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        return 2
    elif x in [12,13, 14, 15, 16]:
        return 1
    else:
        return 3
def manhattan_distance(lat1, lng1, lat2, lng2):
    a = np.abs(lat2 -lat1)
    b = np.abs(lng1 - lng2)
    return a + b
#Bearing
def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))
#Haversine distance
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h
def test(classifier):
    datasettest = pd.read_csv('test.csv')
    x = cleanData(datasettest, ['pickup_time', 'drop_time', 'pickup_year', 'drop_year', 'pickup_month', 'drop_month','drop_hour'])
    x_test = x.iloc[:, 1:27].values
    y_test = classifier.predict(x_test)
    csvfile = open('XGBClassifier5.csv', 'w', newline='')
    fields = ["tripid", "prediction"]
    obj = csv.DictWriter(csvfile, fieldnames=fields)
    obj.writeheader()
    # print(x['tripid'][0],x_test[0],y_test[0],len(y_test),len(x))
    for i in range(len(x)):
        rowrecord = {}
        rowrecord["tripid"] = x['tripid'][i]
        rowrecord["prediction"] = y_test[i]
        obj.writerow(rowrecord)


dataset = pd.read_csv('train.csv')
map_target = {"incorrect": 0, "correct": 1}
dataset['label']=dataset['label'].map(map_target)
y = dataset['label']
X = cleanData(dataset, ['pickup_time', 'drop_time', 'label', 'pickup_year', 'drop_year', 'pickup_month', 'drop_month','drop_hour'])
X = X.iloc[:, 1:27].values
print( X[4])

#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#imputer = imputer.fit(X[:, 0:20])
#X[:, 0:20] = imputer.transform(X[:, 0:20])
# fit model no training data
ftwo_scorer = make_scorer(fbeta_score, average='macro',beta=1)
gnb =XGBClassifier(learning_rate =0.15, n_estimators=300, max_depth=9,
 min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.75,
objective='binary:logistic', nthread=4,  seed=27,reg_alpha= 1,
eval_metric=["auc","error"],max_delta_step=0,tree_method='approx',reg_lambda=1.0,sketch_eps= 0.001,validate_parameters=True)
#subsample=0.85, colsample_bytree=0.75 rate=0.3 depyh 9
#gnb=BaggingClassifier(n_estimators=150,random_state=27)
skf = StratifiedKFold(n_splits=10,random_state=1020, shuffle=True)
#skf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores=[]
mscores=[]
for train_index, test_index in skf.split(X, y):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    gnb.fit(X_train, y_train,early_stopping_rounds=100, eval_set=[(X_test, y_test)],verbose=True)
    #gnb.fit(X_train, y_train)
    f1 = fbeta_score(y_train, gnb.predict(X_train) , average='macro', beta=1)
    y_pred2 = gnb.predict(X_test)

    f2=fbeta_score(y_test, y_pred2, average='macro', beta=1)
    print("Performance sur le train : ", f1)
    print("Performance sur le test : ", f2)
    cm1 = confusion_matrix(y_test, y_pred2)

    print("confusion matrix/n")
    # probs=gnb.predict_proba(X_test)
    # probs = probs[:, 1]
    print(cm1)

    print("/n classification report /n")
    target_names = ['0', '1']

    print(classification_report(y_test, y_pred2, target_names=target_names))

    scores.append(fbeta_score(y_test, y_pred2, average='macro', beta=1))

    mscores.append(matthews_corrcoef(y_test, y_pred2))

print(np.mean(scores))

print(np.mean(mscores))
test(gnb)
