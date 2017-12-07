import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, recall_score, precision_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# from imblearn.over_sampling import SMOTE

def create_features(df, training=True):
    if training:
        df.columns = ['id','last_donation_months','num_donations','volume_donated','first_donation_months', 'march_donation']
    else:
        df.columns = ['id','last_donation_months','num_donations','volume_donated','first_donation_months']
    df['interaction_last_month_num_donations'] =  df['last_donation_months']*df['num_donations']
    df.drop(['volume_donated'],axis=1, inplace=True)
    df['donation_length'] = df['first_donation_months']-df['last_donation_months']
    df['num_donations'].astype(float)
    df['avg_donations_per_month'] = df['num_donations'] /  df['donation_length']
    df['avg_donations_per_month'] = df['avg_donations_per_month'].replace(np.inf, 1)
    df['avg_wait_between_donations'] = df['donation_length'] / df['num_donations']
    df["last_donation_grtr_35"] = np.where(df.last_donation_months > 35,1,0)
    return  df

def get_test_data():
    df = pd.read_csv('data/test.csv')
    df = create_features(df, training=False)
    df.set_index(['id'], inplace=True)
    return df

def get_training_data():
    df = pd.read_csv('data/training.csv')
    df = create_features(df)
    df.set_index(['id'], inplace=True)
    X = df.drop(['march_donation'], axis=1)
    y = df['march_donation']
    return X, y

def find_best_models(models, X, y):
    results = dict()
    for key, values in models.iteritems():
        print 'grid search for {}'.format(key)
        model = values[0]
        params = values[1]
        gridsearch = GridSearchCV(model, params, cv=4,
                                scoring=make_scorer(log_loss, greater_is_better=False,
                                                    needs_proba=True, needs_threshold=False))
        gridsearch.fit(X, y)
        res = [gridsearch.best_estimator_, gridsearch.best_score_]
        results[key] = res
    return results

def save_submission(IDs, donate_probs):
    f = open('submission.csv', "w")
    f.write(",Made Donation in March 2007\n")
    for ID, prob in zip(IDs, donate_probs):
        f.write("{},{}\n".format(ID,prob[0]))
    f.close()

if __name__ == '__main__':
    X, y = get_training_data()
    X_testing = get_test_data()
    print 'length of test data: {}'.format(len(X_testing))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # sm = SMOTE(random_state=42)
    # X_res, y_res = sm.fit_sample(X, y)
    # X_train, y_train = X_res, y_res
    #
    # model_list = {'rfc':[RandomForestClassifier(n_jobs=-1, oob_score=True),{'n_estimators': [50, 100, 150,300],
    #                                                                         'max_depth': [5, 10 ,15],
    #                                                                         'max_features': [2,4,6]}],
    #               'gbc':[GradientBoostingClassifier(),{'learning_rate': [0.1, 0.01,0.005,0.001],
    #                                                     'n_estimators': [50,100,200,250],
    #                                                     'subsample': [1.0,0.8],
    #                                                     'max_features': [2,4],
    #                                                     'max_depth': [3,5]}],
    #               'mlp':[MLPClassifier(early_stopping=True),{'hidden_layer_sizes': [(500,),(500,2)],
    #                                                          'activation': ['relu','logistic'],
    #                                                          'learning_rate_init': [0.001, 0.0005],
    #                                                          'max_iter':[200,400]}]}
    model_list = {'mlp':[MLPClassifier(),{'hidden_layer_sizes': [(500,),(500,2),(700,)],
                                               'activation': ['relu','logistic'],
                                               'learning_rate_init': [0.001, 0.0005],
                                               'max_iter':[200,400],
                                               'batch_size': [100,200,300]}]}

    best_models = find_best_models(model_list, X_train, y_train)
    best_name = max(best_models.iterkeys(), key=(lambda key: best_models[key][1]))
    model = best_models[best_name][0]
    print 'Best score from cross_validation: {}'.format(best_models[best_name][1])
    model.fit(X_train, y_train)
    y_test_probs = model.predict_proba(X_test)[:,1]
    loss = log_loss(y_test, y_test_probs)
    print 'log_loss: {}'.format(loss)


    y_test_predict = model.predict_proba(X_testing)[:,1]
    IDs = X_testing.index.values
    save_submission(IDs, y_test_predict)
