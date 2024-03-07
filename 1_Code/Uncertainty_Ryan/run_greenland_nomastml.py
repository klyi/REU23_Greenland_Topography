import pandas as pd
from mastml.mastml import Mastml
from mastml.datasets import LocalDatasets
from mastml.preprocessing import SklearnPreprocessor
from mastml.models import SklearnModel, EnsembleModel
from mastml.data_splitters import SklearnDataSplitter, NoSplit, LeaveMultiGroupOut
from mastml.feature_selectors import EnsembleModelFeatureSelector, NoSelect
from mastml.feature_generators import OneHotGroupGenerator

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
import pandas as pd

mastml = Mastml(savepath='results/greenland_nomastml')
savepath = mastml.get_savepath
mastml_metadata = mastml.get_mastml_metadata

target = 'track_bed_target'

extra_columns = ['surf_x', 'surf_y', 'track_bed_x', 'track_bed_y', 'index']

d = LocalDatasets(file_path='mastml/data/data_nested_subset.csv',
                  target=target,
                  extra_columns=extra_columns,
                  group_column=None, #fluence_group
                  testdata_columns=None,
                  as_frame=True)

data_dict = d.load_data(copy=True, savepath=savepath)
X = data_dict['X']
y = data_dict['y']
X_extra = data_dict['X_extra']
groups = data_dict['groups']
X_testdata = data_dict['X_testdata']

output_dict = dict()

rkf = RepeatedKFold(n_repeats=5, n_splits=5)
for i, (train_index, test_index) in enumerate(rkf.split(X)):
    print('On fold', i)
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]

    print('Scaling')
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)

    print('Fitting')
    rfr = RandomForestRegressor(n_estimators=250)
    #rfr = XGBRegressor(n_estimators=350, max_depth=7, min_child_weight=0.25, subsample=0.8, eta=0.25)
    rfr.fit(X_train, y_train)

    print('Predicting')
    y_pred = rfr.predict(X_test)
    
    output_dict[i] = dict()
    output_dict[i]['y_test'] = np.array(y_test).tolist()
    output_dict[i]['y_pred'] = np.array(y_pred).tolist()

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_std = rmse/np.std(y_test)

    output_dict[i]['r2'] = r2
    output_dict[i]['mae'] = mae
    output_dict[i]['rmse'] = rmse
    output_dict[i]['rmse_std'] = rmse_std

    #'''
    X_aslist = X_test.tolist()
    print('Error bars')
    ebars = list()
    for x in range(len(X_aslist)):
        preds = list()
        for pred in rfr.estimators_:
            preds.append(pred.predict(np.array(X_aslist[x]).reshape(1, -1))[0])
        ebars.append(np.std(preds))

    output_dict[i]['ebars'] = ebars
    #'''

    print('Saving split')
    df = pd.DataFrame.from_dict(output_dict, orient='index').T
    df.to_csv(os.path.join(savepath, 'Greenland_data_output_fold_'+str(i)+'.csv'))

