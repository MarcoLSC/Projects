import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats as sm
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.datasets import load_iris

data = pd.read_stata(r"ESS_data/ESS7e02_2.dta", convert_categoricals=False) #removed clean version for prod


# Data cleaning
data = data[(data['fltdpr'] > 0) & (data['fltdpr'] <= 4)]
data = data[(data['gndr'] < 3) & (data['gndr'] > 0)]
data = data[(data['hhmmb'] < 30) & (data['hhmmb'] > 0)]
data = data[(data['edulvlb'] < 5556) & (data['edulvlb'] > 1)]
data = data[(data['maritalb'] < 6) & (data['maritalb'] > 0)]
data = data[(data['hincfel'] <= 4) & (data['hincfel'] >= 1)]
data = data[(data['hlthhmp'] <= 3) & (data['hlthhmp'] >= 1)]
data = data[(data['fltsd'] <= 4) & (data['fltsd'] >= 1)]


# New vars generation
data["anweight"] = data["pspwght"] * data["pweight"]

data["agea2"] = data["agea"] * data["agea"]

data['unemployed'] = 0
data.loc[(data['uempla'] == 1) | (data['uempli'] == 1), 'unemployed'] = 1

data['health_hampered'] = 0
data.loc[(data['hlthhmp'] < 3), 'health_hampered'] = 1

data['inthr'] = 0
data.loc[(data['inwshh'] >= 8) & (data['inwshh'] <= 13), 'inthr'] = 1
data.loc[(data['inwshh'] >= 14) & (data['inwshh'] <= 18), 'inthr'] = 2
data.loc[(data['inwshh'] >= 19) & (data['inwshh'] <= 22), 'inthr'] = 3

data['depressed'] = 0
data.loc[(data['fltdpr'] >= 3), 'depressed'] = 1

# Cerating descriptive statistics table

df_table = data[["fltdpr",
              "hinctnta",
              "gndr",
              "agea",
              "hhmmb",
              "edulvlb",
             "uempla",
              "uempli", 
              "unemployed",
              "hincfel",
              "health_hampered",
              "cntry",
              "eduyrs",
              "inwmms",
              "inwyys",
              "inthr",
              "rtrd"
             ]].copy()

summary_table = np.round(sm.descriptivestats.describe(df_table), 4).T[['nobs','mean', 'std','median', 'min', 'max']]

with open('descriptive_table.tex','w') as tf:
    tf.write(summary_table.to_latex(
        header=['Count','Mean','St. deviation','Median','Min','Max'],
        caption='Descriptive statistics of most relevant variables')
            )


# List of variables that need dummies
var_dummies = ['edulvlb',
               'maritalb',
               'hincfel',
               'cntry',
               'inwmms',
               'inwyys',
               'inthr',
               'health',
               'nacer2',
               'idno'
              ]

# Keeping only the relevant variables
rdata = data[["fltdpr",
              "hinctnta",
              "gndr",
              "agea",
              "agea2",
              "hhmmb",
              "edulvlb",
              "maritalb",
              "unemployed",
              "hincfel",
              "health_hampered",
              "cntry",
              "eduyrs",
              "inwmms",
              "inwyys",
              "inthr",
              "anweight",
              "health",
              "chldhm",
              "uempla",
              "uempli", 
              "nacer2",
              "idno", 
              "rtrd", 
              "dvrcdeva"
             ]].copy()

X = rdata[["hinctnta",
           "gndr",
           "agea",
           "agea2",
           "hhmmb",
           "edulvlb",
           "maritalb",
           "unemployed",
           "rtrd",
           "hincfel",
           "health_hampered",
           "cntry",
           "eduyrs",
           "inwmms",
           "inwyys",
           "inthr"
          ]]

weights = rdata['anweight']

# Code for adding dummies for some variables to the whole dataset
train_x = pd.get_dummies(X, columns=['edulvlb','maritalb','hincfel', 'cntry', 'inwmms', 'inwyys', 'inthr'])

# Regression part - turned into a function for prod
def multilogit(regr_data, X_list, X_to_encode, y_col_name, weights_col_name='anweight', baseoutcome=False):
    rdata = regr_data[X_list + [y_col_name, weights_col_name]].copy()
    if X_to_encode:
        rdata = pd.get_dummies(rdata, columns=X_to_encode, drop_first=True)
        generated_dummies = [i for i in rdata.columns if "_" in i]
        X_list = X_list + generated_dummies
        X_list = [x for x in X_list if x not in X_to_encode]
    rdata.dropna(axis=0, how='any', inplace=True)
    y = rdata[y_col_name]
    noobs=len(y)
    print(rdata.columns)
    X = rdata[X_list]
    weights = rdata[weights_col_name]
    clf = linear_model.LogisticRegression(multi_class='multinomial', max_iter=8000, C = 1e9, solver='newton-cg').fit(X, y, weights)

    if not baseoutcome:
        adj_intercept = pd.Series(clf.intercept_)
        adj_coef = pd.DataFrame(clf.coef_)
    else:
        baseoutcome_index = list(set(y)).index(baseoutcome)
        adj_intercept = pd.Series(clf.intercept_ - clf.intercept_[baseoutcome_index])
        adj_coef = pd.DataFrame(clf.coef_ - clf.coef_[baseoutcome_index])

    adj_coef.columns = list(X.columns)
    adj_coef["Outcome"] = list(set(y))
    adj_coef["Intercept"] = adj_intercept
    adj_coef = adj_coef[['Outcome', 'Intercept'] + [c for c in adj_coef if c not in ['Outcome', 'Intercept']]]
    """
    y_hat = clf.predict_proba(X)
    y_hat.columns = ["p1", "p2", "p3", "p4"]
    print(rdata)
    print(rdata.shape)
    print("yhat is", y_hat)
    print(y_hat.shape)
    rdata["y_hat"] = y_hat
    for i in list(set(y)):
        print(i)
        sdata = rdata[rdata[y_col_name] == i].copy()
        sX = sdata[X_list]
        sy = sdata[y_col_name]
        sy_hat = sdata["y_hat"]
        #return [sy, sy_hat]
        snoobs = len(sy)
        #X = rdata[X_list]
        residuals = sy.values - sy_hat
        residual_sum_of_squares = residuals.T @ residuals
        sigma_squared_hat = residual_sum_of_squares / (snoobs - len(sX) - 1)
        sX_with_intercept = np.hstack([np.ones((sX.shape[0], 1)), sX])
        var_beta_hat = np.linalg.inv(sX_with_intercept.T @ sX_with_intercept) * sigma_squared_hat
        print(var_beta_hat)
        for p_ in range(len(sX)):
            standard_error = var_beta_hat[p_, p_] ** 0.5
            print(f"SE(beta_hat[{p_}]): {standard_error}")
        print("end of",i)
    """
    return {"Model:" : clf,
            "No. of obs:" : noobs,
            "Coefficients" : adj_coef}

a = multilogit(data, ["hinctnta", "rtrd", "agea", 'hhmmb'], [],
               "fltdpr", 'anweight', baseoutcome=1)

print(a)
print("Starting regr b")

baseline_no_hhmp = multilogit(data,
               ["hinctnta",
                "gndr",
                "agea",
                "agea2",
                "hhmmb",
                "edulvlb",
                "maritalb",
                "unemployed",
                "rtrd",
                "hincfel",
                "cntry",
                "eduyrs",
                "inwmms",
                "inwyys",
                "inthr"],
               ['edulvlb',
               'maritalb',
               'hincfel',
               'cntry',
               'inwmms',
               'inwyys',
               'inthr'],
               "fltdpr",
               'anweight',
               baseoutcome=1)

print(baseline_no_hhmp["Coefficients"])

baseline_with_hhmp = multilogit(data,
               ["hinctnta",
                "gndr",
                "agea",
                "agea2",
                "hhmmb",
                "edulvlb",
                "maritalb",
                "unemployed",
                "rtrd",
                "hincfel",
                "health_hampered",
                "cntry",
                "eduyrs",
                "inwmms",
                "inwyys",
                "inthr"],
               ['edulvlb',
               'maritalb',
               'hincfel',
               'cntry',
               'inwmms',
               'inwyys',
               'inthr'],
               "fltdpr",
               'anweight',
               baseoutcome=1)

print(baseline_with_hhmp["Coefficients"])
