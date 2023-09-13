"""
SapientML assigns some internal labels for each ML component. This file has the information of those labels.
"""


name_to_label_mapping = {
    "random forest": {
        "c": "MODEL:Classifier:RandomForestClassifier:sklearn",
        "r": "MODEL:Regressor:RandomForestRegressor:sklearn",
    },
    "extra tree": {
        "c": "MODEL:Classifier:ExtraTreesClassifier:sklearn",
        "r": "MODEL:Regressor:ExtraTreesRegressor:sklearn",
    },
    "lightgbm": {"c": "MODEL:Classifier:LGBMClassifier:lightgbm", "r": "MODEL:Regressor:LGBMRegressor:lightgbm"},
    "xgboost": {"c": "MODEL:Classifier:XGBClassifier:xgboost", "r": "MODEL:Regressor:XGBRegressor:xgboost"},
    "catboost": {
        "c": "MODEL:Classifier:CatBoostClassifier:catboost",
        "r": "MODEL:Regressor:CatBoostRegressor:catboost",
    },
    "gradient boosting": {
        "c": "MODEL:Classifier:GradientBoostingClassifier:sklearn",
        "r": "MODEL:Regressor:GradientBoostingRegressor:sklearn",
    },
    "adaboost": {"c": "MODEL:Classifier:AdaBoostClassifier:sklearn", "r": "MODEL:Regressor:AdaBoostRegressor:sklearn"},
    "decision tree": {
        "c": "MODEL:Classifier:DecisionTreeClassifier:sklearn",
        "r": "MODEL:Regressor:DecisionTreeRegressor:sklearn",
    },
    "svm": {"c": "MODEL:Classifier:SVC:sklearn", "r": "MODEL:Regressor:SVR:sklearn"},
    "linear svm": {"c": "MODEL:Classifier:LinearSVC:sklearn", "r": "MODEL:Regressor:LinearSVR:sklearn"},
    "logistic/linear regression": {
        "c": "MODEL:Classifier:LogisticRegression:sklearn",
        "r": "MODEL:Regressor:LinearRegression:sklearn",
    },
    "lasso": {"r": "MODEL:Regressor:Lasso:sklearn"},
    "sgd": {"c": "MODEL:Classifier:SGDClassifier:sklearn", "r": "MODEL:Regressor:SGDRegressor:sklearn"},
    "mlp": {"c": "MODEL:Classifier:MLPClassifier:sklearn", "r": "MODEL:Regressor:MLPRegressor:sklearn"},
    "multinomial nb": {"c": "MODEL:Classifier:MultinomialNB:sklearn"},
    "gaussian nb": {"c": "MODEL:Classifier:GaussianNB:sklearn"},
    "bernoulli nb": {"c": "MODEL:Classifier:BernoulliNB:sklearn"},
}


def map_label_to_name():
    label_to_name = {"MODEL:Classifier:LGBMClassifier:lgbm": "lightgbm", "MODEL:Regressor:train:xgboost": "xgboost"}
    for k, v in name_to_label_mapping.items():
        for k1, v1 in v.items():
            label_to_name[v1] = k
    return label_to_name
