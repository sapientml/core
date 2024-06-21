# Copyright 2023-2024 The SapientML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ast
import os

from sapientml.util.logging import setup_logger

logger = setup_logger()

"""
    I edited khsamaha_aviation-accident-database-synopses/predict-percentage-of-fatal-injuries.py
    since the original one is not complete and cannot be parsed
"""


model_match_keywords = [
    ("RandomForestRegressor", "random forest", "regression"),
    ("RandomForestClassifier", "random forest", "classification"),
    ("BaggingClassifier", "bagging", "classification"),
    ("BaggingRegressor", "bagging", "regression"),
    ("AdaBoostRegressor", "adaboost", "regression"),
    ("LinearRegression", "logistic/linear regression", "regression"),
    ("LogisticRegression", "logistic/linear regression", "classification"),
    ("SVC", "svm", "classification"),
    ("SVR", "svm", "regression"),
    ("MultinomialNB", "multinomial nb", "classification"),
    ("GaussianNB", "gaussian nb", "classification"),
    ("XGBClassifier", "xgboost", "classification"),
    ("XGBRegressor", "xgboost", "regression"),
    ("xgboost", "xgboost", "classification"),
    ("DecisionTreeClassifier", "decision tree", "classification"),
    ("DecisionTreeRegressor", "decision tree", "regression"),
    ("Ridge", "ridge", "regression"),
    ("KNeighborsClassifier", "knn", "classification"),
    ("Lasso", "lasso", "regression"),
    ("LinearSVC", "linear svm", "classification"),
    ("SVC(kernel='linear')", "linear svm", "classification"),
    # view Gaussian family GLM as ridge regressor
    ("GLM", "ridge", "regression"),
    ("OLS", "logistic/linear regression", "regression"),
    ("GradientBoostingClassifier", "gradient boosting", "classification"),
    ("GradientBoostingRegressor", "gradient boosting", "regression"),
    ("ExtraTreesRegressor", "extra tree", "regression"),
    ("LGBMClassifier", "lightgbm", "classification"),
    ("Logit", "logistic/linear regression", "classification"),
    ("MLPClassifier", "mlp", "classification"),
    # ad-hoc
    ("RFE", "logistic/linear regression", "classification"),
    # new
    ("KMeans", "kmeans", "classification"),
    ("ElasticNet", "logistic/linear regression", "regression"),
    ("MLPRegressor", "mlp", "regression"),
    ("AdaBoostClassifier", "adaboost", "classification"),
    ("SGDRegressor", "sgd", "regression"),
    ("SGDClassifier", "sgd", "classification"),
    ("ExtraTreesClassifier", "extra tree", "classification"),
    ("Perceptron", "sgd", "classification"),
    ("LGBMRegressor", "lightgbm", "regression"),
    ("KNeighborsRegressor", "knn", "regression"),
]

# only GridSearchCV is hard to detect model type
# only RandomizedSearchCV is hard to mutate
# only VotingClassifier is hard to mutate
# only BaggingClassifier is hard to mutate

# TODO:
#  need to handle xgboost's DMatrix (minor: only affect one dataset)


order = [
    "random forest",
    "extra tree",
    "lightgbm",
    "xgboost",
    "catboost",
    "gradient boosting",
    "adaboost",
    "decision tree",
    "svm",
    "linear svm",
    "logistic/linear regression",
    "lasso",
    "sgd",
    "ridge",
    "knn",
    "bagging",  # new
    "mlp",
    "multinomial nb",
    "gaussian nb",
    "bernoulli nb",
]

# not alter the prediction statement now
templates = {
    # 'model type': (definition statement, training statement)
    ("random forest", "c"): (
        """from sklearn.ensemble import RandomForestClassifier
#1 = RandomForestClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("random forest", "r"): (
        """from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(RandomForestRegressor())
else:
    #1 = RandomForestRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("decision tree", "c"): (
        """from sklearn.tree import DecisionTreeClassifier
#1 = DecisionTreeClassifier(random_state=0)""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("decision tree", "r"): (
        """from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(DecisionTreeRegressor(random_state=0))
else:
    #1 = DecisionTreeRegressor(random_state=0)""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("extra tree", "c"): (
        """from sklearn.ensemble import ExtraTreesClassifier
#1 = ExtraTreesClassifier(random_state=0)""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("extra tree", "r"): (
        """from sklearn.ensemble import ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(ExtraTreesRegressor(random_state=0))
else:
    #1 = ExtraTreesRegressor(random_state=0)""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    # =================================
    ("xgboost", "c"): (
        """import xgboost as xgb
#1 = xgb.XGBClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("xgboost", "r"): (
        """import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(xgb.XGBRegressor())
else:
    #1 = xgb.XGBRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("catboost", "c"): (
        """import catboost as cat
#1 = cat.CatBoostClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("catboost", "r"): (
        """import catboost as cat
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(cat.CatBoostRegressor())
else:
    #1 = cat.CatBoostRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("gradient boosting", "c"): (
        """from sklearn.ensemble import GradientBoostingClassifier
#1 = GradientBoostingClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("gradient boosting", "r"): (
        """from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(GradientBoostingRegressor())
else:
    #1 = GradientBoostingRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("adaboost", "c"): (
        """from sklearn.ensemble import AdaBoostClassifier
#1 = AdaBoostClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("adaboost", "r"): (
        """from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(AdaBoostRegressor())
else:
    #1 = AdaBoostRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("lightgbm", "c"): (
        """from lightgbm import LGBMClassifier
#1 = LGBMClassifier()""",
        """import pandas
if isinstance(#XT, pandas.DataFrame): #XT.columns = [str(i) for i in range(len(#XT.columns))]
#2 = #1.fit(#XT, #YT)""",
        """import pandas
if isinstance(#XS, pandas.DataFrame): #XS.columns = [str(i) for i in range(len(#XS.columns))]""",
    ),
    ("lightgbm", "r"): (
        """from lightgbm import LGBMRegressor
import pandas
from sklearn.multioutput import MultiOutputRegressor
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(LGBMRegressor())
else:
    #1 = LGBMRegressor()""",
        """import pandas
if isinstance(#XT, pandas.DataFrame):
    #XT.columns = [str(i) for i in range(len(#XT.columns))]
#2 = #1.fit(#XT, #YT)""",
        """import pandas
if isinstance(#XS, pandas.DataFrame):
    #XS.columns = [str(i) for i in range(len(#XS.columns))]""",
    ),
    # =================================
    ("svm", "c"): (
        """from sklearn.svm import SVC
#1 = SVC()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("svm", "r"): (
        """from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(SVR())
else:
    #1 = SVR()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("linear svm", "c"): (
        """from sklearn.svm import LinearSVC
#1 = LinearSVC()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("linear svm", "r"): (
        """from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(LinearSVR())
else:
    #1 = LinearSVR()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("logistic/linear regression", "c"): (
        """from sklearn.linear_model import LogisticRegression
#1 = LogisticRegression()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("logistic/linear regression", "r"): (
        """from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(LinearRegression())
else:
    #1 = LinearRegression()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("lasso", "r"): (
        """from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(Lasso())
else:
    #1 = Lasso()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("sgd", "c"): (
        """from sklearn.linear_model import SGDClassifier
#1 = SGDClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("sgd", "r"): (
        """from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(SGDRegressor())
else:
    #1 = SGDRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    # =================================
    ("mlp", "c"): (
        """from sklearn.neural_network import MLPClassifier
#1 = MLPClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("mlp", "r"): (
        """from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(MLPRegressor())
else:
    #1 = MLPRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    # =================================
    ("multinomial nb", "c"): (
        """from sklearn.naive_bayes import MultinomialNB
#1 = MultinomialNB()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("gaussian nb", "c"): (
        """from sklearn.naive_bayes import GaussianNB
#1 = GaussianNB()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("bernoulli nb", "c"): (
        """from sklearn.naive_bayes import BernoulliNB
#1 = BernoulliNB()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    # ===================================
    ("ridge", "c"): (
        """from sklearn.linear_model import RidgeClassifier
#1 = RidgeClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("ridge", "r"): (
        """from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(Ridge())
else:
    #1 = Ridge()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("knn", "c"): (
        """from sklearn.neighbors import KNeighborsClassifier
#1 = KNeighborsClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("knn", "r"): (
        """from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(KNeighborsRegressor())
else:
    #1 = KNeighborsRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("bagging", "c"): (
        """from sklearn.ensemble import BaggingClassifier
#1 = BaggingClassifier()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
    ("bagging", "r"): (
        """from sklearn.ensemble import BaggingRegressor
from sklearn.multioutput import MultiOutputRegressor
import pandas
if isinstance(#YT, pandas.DataFrame) and len(#YT.columns) > 1:
    #1 = MultiOutputRegressor(BaggingRegressor())
else:
    #1 = BaggingRegressor()""",
        "#2 = #1.fit(#XT, #YT)",
    ),
}


def _catch_target_name(assign_stmt: ast.Assign):
    # only digest the first node
    target = assign_stmt.targets[0]

    if isinstance(target, ast.Name):
        return target.id
    else:
        for node_name in ast.walk(target):
            if isinstance(node_name, ast.Name):
                return node_name.id


def _catch_func_name(call_stmt):
    if not isinstance(call_stmt, ast.Call):
        return None
    else:
        func = call_stmt.func
        if isinstance(func, ast.Name):
            func = func.id
        elif isinstance(func, ast.Attribute):
            func = func.attr
        elif isinstance(func, ast.Lambda):
            func = None
        return func


def _catch_method_varname(node):
    if isinstance(node, ast.Name):
        return node.id, list()
    elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return ast.unparse(node), list()
    for nownode in ast.walk(node):
        if isinstance(nownode, ast.Call) and ast.unparse(nownode.func) not in [
            "print"
        ]:  # filter out useless function call
            method_name = ast.unparse(nownode.func)
            variable_names = [ast.unparse(arg_node) for arg_node in nownode.args] + [
                ast.unparse(kw_node.value) for kw_node in nownode.keywords
            ]
            return method_name, variable_names
    return None, list()


class MutationException(Exception):
    """MutationException class"""

    pass


# init: check for what model types, the templates are missing
for o in order:
    for t in ["c", "r"]:
        if (o, t) not in templates:
            logger.info(f"template for {(o, t)} is missing")


# Parse different entities from pipeline using AST library.
def pipeline_analysis(in_path):
    """Parse different entities from pipeline using AST library.

    Parameters
    ----------
    in_path: PosixPath
        It contains path from pathlib.PosixPath

    Returns
    -------
        m_type : str
            It returns a string variable.
        task_type : str
            It returns a string variable.
        def_stmt : tuple
            It returns a tuple.
        train_stmt : tuple
            It returns a tuple.
        test_stmt : tuple
            It returns a tuple.
        model_pred_var : str
            It returns a string variable.
        model_train_var : str
            It returns a string variable.
        train_X.strip() : str
            It returns a string variable.
        train_Y.strip() : str
            It returns a string variable.
        test_X.strip() :  str
            It returns a string variable.
    """
    with open(in_path, "r", encoding="utf-8") as f:
        original_script = f.read()

    tree = ast.parse(source=original_script, filename=in_path)  # parse pipeline to tree

    possible_training_stmts = list()
    possible_test_stmts = list()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            func = node.func
            if isinstance(func, ast.Call):
                func = func.func
            if isinstance(func, ast.Name):
                func = func.id
            elif isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    name = func.value.id
                elif isinstance(func.value, ast.AST):
                    for node_name in ast.walk(func.value):
                        if isinstance(node_name, ast.Name):
                            name = node_name.id
                else:
                    raise Exception("Should not happen...")

                func = func.attr
            elif isinstance(func, ast.Lambda):
                continue
            else:
                raise Exception("Unknown func in call stmt", ast.unparse(func), ast.dump(func))
            # support xgb.train is non-trivial, and has benefit only on one notebook: hubh0799_churn-modelling/churn-modelling-xgboost-aucroc-0-902347.py
            if func in ["fit"]:
                possible_training_stmts.append((name, func, node.lineno))
            if func in ["predict", "predict_proba"]:
                possible_test_stmts.append((name, func, node.lineno))

    if len(possible_training_stmts) == 0 or len(possible_test_stmts) == 0:
        raise MutationException("No training or test statement found")
    else:
        pass

    # choose the last possible testing statement as the testing statement
    test_stmt = sorted(possible_test_stmts, key=lambda x: x[2], reverse=True)[0]
    train_stmt = None
    def_stmt = None
    model_pred_var = model_train_var = test_stmt[0]

    # determine real training statement and model definition statement
    possible_training_line_no = [x[2] for x in possible_training_stmts]
    import_name_mapping = dict()
    def_stmts = dict()
    for node in tree.body:
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            names = node.names
            for a in names:
                if a.asname is not None:
                    import_name_mapping[a.asname] = a.name
        if isinstance(node, ast.Assign):
            if node.lineno in possible_training_line_no:
                now_assigned_var = _catch_target_name(node)
                # if the assigned one matches the predict var name, then proceed
                if now_assigned_var == test_stmt[0]:
                    now_train_var = [x[0] for x in possible_training_stmts if x[2] == node.lineno][0]
                    if now_train_var not in def_stmts:
                        # means the training statement uses a in-place defined model
                        # in this case, we view the training var name as the same as the predict var name
                        train_stmt = [x for x in possible_training_stmts if x[2] == node.lineno][0]
                        train_stmt = (model_train_var, train_stmt[1], train_stmt[2])
                        def_stmt = (model_train_var, train_stmt[1], train_stmt[2])
                    else:
                        model_train_var = now_train_var
                        train_stmt = [x for x in possible_training_stmts if x[2] == node.lineno][0]
                        def_stmt = def_stmts[now_train_var]

            now_target = _catch_target_name(node)
            # if now_target not in def_stmts:
            def_stmts[now_target] = (now_target, _catch_func_name(node.value), node.lineno)

        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if node.lineno in possible_training_line_no:
                now_train_stmt = [x for x in possible_training_stmts if x[2] == node.lineno][0]
                if now_train_stmt[0] == model_train_var:
                    train_stmt = now_train_stmt
                    def_stmt = def_stmts[model_train_var]

    if def_stmt is None or train_stmt is None or test_stmt is None:
        raise MutationException("Unable to locate def stmt or train stmt or test stmt")

    # relocate the test stmt as the closest immediate one after training stmt
    test_stmt = sorted([t for t in possible_test_stmts if t[2] >= train_stmt[2]], key=lambda x: x[2])[0]

    # extract method name and variables in definition statement and training statement
    for node in tree.body:
        if node.lineno == def_stmt[2]:
            dnode = node
            if def_stmt[2] == train_stmt[2]:
                # an inplace definition statement
                # in this case, we need to trim the root calling part
                if isinstance(dnode, ast.Assign) and isinstance(dnode.value, ast.Call):
                    dnode = dnode.value.func
            def_method_name, def_var_names = _catch_method_varname(dnode)
            if def_method_name is None:
                raise MutationException(f"Unusual model definition statement: {ast.unparse(node)}")

        if node.lineno == train_stmt[2]:
            train_method_name, train_var_names = _catch_method_varname(node)

    for call_node in ast.walk(tree):
        if (
            isinstance(call_node, ast.Call)
            and isinstance(call_node.func, ast.Attribute)
            and call_node.func.attr in ["predict", "predict_proba"]
        ):
            test_method_name, test_var_names = _catch_method_varname(call_node)

    # determine model type from definition statement
    model_signature = (
        (def_method_name + " " + " ".join(def_var_names))
        .replace(".", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace(",", " ")
        .split(" ")
    )
    tmp = list()
    for item in model_signature:
        for k, w in import_name_mapping.items():
            if k == item:
                item = w
        tmp.append(item)
    model_signature = " ".join(tmp)

    kws, m_types, task_types = list(), list(), list()
    for kw, m_type, task_type in model_match_keywords:
        if model_signature.count(kw) > 0:
            kws.append(kw)
            m_types.append(m_type)
            task_types.append(task_type)
    if len(kws) > 1 and "xgboost" in kws:
        # "xgboost" has lower priority than other keywords
        index_of = kws.index("xgboost")
        del kws[index_of]
        del m_types[index_of]
        del task_types[index_of]
    if len(kws) == 2 and "SVC" in kws and ("LinearSVC" in kws or "SVC(kernel='linear')" in kws):
        kws = ["LinearSVC"]
        m_types = ["linear svm"]
        task_types = ["classification"]
    if len(kws) > 1 and "adaboost" in m_types:
        # adaboost has higher priority than other keywords
        index_of = m_types.index("adaboost")
        kws = [kws[index_of]]
        m_types = [m_types[index_of]]
        task_types = [task_types[index_of]]

    if (
        model_signature.count("GridSearch") > 0
        or model_signature.count("GridSearchCV") > 0
        or model_signature.count("VotingClassifier") > 0
        or model_signature.count("VotingRegressor") > 0
        or model_signature.count("RandomizedSearchCV") > 0
        or model_signature.count("Pipeline") > 0
        or model_signature.count("make_pipeline")
    ):
        if len(kws) != 1:
            raise MutationException(
                "For gridsearch, cross validation, pipeline, and voting classifier, without more information we cannot determine the model type"
            )
    else:
        if len(kws) == 0:
            raise MutationException(f"Unsupport model type: {def_method_name}")
        else:
            assert len(kws) == 1

    # model type
    kw = kws[0]
    m_type = m_types[0]
    task_type = task_types[0]
    if kw == "GLM" or kw == "OLS" or kw == "Logit":
        # for models in statsmodels, the order of training and test datasets are reversed
        # and training and test datasets are defined in the model definition statement
        train_X, train_Y = def_var_names[1], def_var_names[0]
    else:
        if len(train_var_names) < 2:
            raise MutationException(f"Only {len(train_var_names)} train variable.")
        train_X, train_Y = train_var_names[0], train_var_names[1]
        if len(train_var_names) == 0:
            raise MutationException("No train variable found.")
    if len(test_var_names) == 0:
        raise MutationException("No train variable found.")
    test_X = test_var_names[0]
    assert task_type in ["classification", "regression"]

    return (
        m_type,
        task_type,
        def_stmt,
        train_stmt,
        test_stmt,
        model_pred_var,
        model_train_var,
        train_X.strip(),
        train_Y.strip(),
        test_X.strip(),
    )


# Creates mutated pipelines with the provided template models.
def model_mutation(in_path, out_dir, gen_mutation=True, gen_df_dumper=True):
    """Creates mutated pipelines with the provided template models.

    Parameters
    ----------
    in_path : PosixPath
        It contains path from pathlib.PosixPath.
    out_dir : PosixPath
        It contains path from pathlib.PosixPath.
    gen_mutation : bool
        True and otherwise False.
    gen_df_dumper : bool
        True and otherwise False.

    Returns
    -------
    m_type : str
        It returns a string.
    muted_types : list
        It returns a list.

    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    (
        m_type,
        task_type,
        def_stmt,
        train_stmt,
        test_stmt,
        model_pred_var,
        model_train_var,
        train_X,
        train_Y,
        test_X,
    ) = pipeline_analysis(in_path)

    with open(in_path, "r", encoding="utf-8") as f:
        original_script = f.read()

    # first, dump current script to corresponding model type folder
    if not os.path.exists(out_dir / m_type):
        os.makedirs(out_dir / m_type)
    with open(out_dir / m_type / "script.py", "w", encoding="utf-8") as f:
        f.write(original_script)

    script_list = original_script.split("\n")

    muted_types = list()

    # then, start the main mutation script
    for key in templates:
        if not isinstance(key, tuple):
            continue
        mut_type, mut_task_type = key
        assert mut_type in order
        if mut_task_type == {"classification": "c", "regression": "r"}[task_type] and mut_type != m_type:
            value = templates[key]
            assert len(value) in [2, 3]
            if len(value) == 2:
                new_def_stmt, new_train_stmt = value
                ins_pred_stmt = None
            else:
                new_def_stmt, new_train_stmt, ins_pred_stmt = value

            if not os.path.exists(out_dir / mut_type):
                os.makedirs(out_dir / mut_type)

            new_script_list = script_list.copy()

            def_to_insert = (
                new_def_stmt.replace("#1", model_train_var)
                .replace("#2", model_pred_var)
                .replace("#XT", train_X)
                .replace("#YT", train_Y)
            )
            train_to_insert = (
                new_train_stmt.replace("#1", model_train_var)
                .replace("#2", model_pred_var)
                .replace("#XT", train_X)
                .replace("#YT", train_Y)
            )
            test_to_insert = None if ins_pred_stmt is None else ins_pred_stmt.replace("#XS", test_X)

            # index starts from 0
            t_def_stmt = list(def_stmt).copy()
            t_def_stmt[2] -= 1
            t_train_stmt = list(train_stmt).copy()
            t_train_stmt[2] -= 1
            t_test_stmt = list(test_stmt).copy()
            t_test_stmt[2] -= 1

            def_indent = (
                ""
                if len(script_list[t_def_stmt[2]].lstrip()) == len(script_list[t_def_stmt[2]])
                else script_list[t_def_stmt[2]][0]
                * (len(script_list[t_def_stmt[2]]) - len(script_list[t_def_stmt[2]].lstrip()))
            )
            train_indent = (
                ""
                if len(script_list[t_train_stmt[2]].lstrip()) == len(script_list[t_train_stmt[2]])
                else script_list[t_train_stmt[2]][0]
                * (len(script_list[t_train_stmt[2]]) - len(script_list[t_train_stmt[2]].lstrip()))
            )
            test_indent = (
                ""
                if len(script_list[t_test_stmt[2]].lstrip()) == len(script_list[t_test_stmt[2]])
                else script_list[t_test_stmt[2]][0]
                * (len(script_list[t_test_stmt[2]]) - len(script_list[t_test_stmt[2]].lstrip()))
            )
            def_to_insert = "\n".join([train_indent + item for item in def_to_insert.split("\n")])
            train_to_insert = "\n".join([def_indent + item for item in train_to_insert.split("\n")])
            test_to_insert = (
                script_list[t_test_stmt[2]]
                if test_to_insert is None
                else "\n".join(
                    [test_indent + item for item in test_to_insert.split("\n")] + [script_list[t_test_stmt[2]]]
                )
            )

            if t_def_stmt[2] == t_train_stmt[2]:
                new_script_list[t_def_stmt[2]] = "\n".join([def_to_insert, train_to_insert])
            else:
                new_script_list[t_def_stmt[2]] = def_to_insert
                new_script_list[t_train_stmt[2]] = train_to_insert
            new_script_list[t_test_stmt[2]] = test_to_insert

            # comment out column name printing lines and other unsupported specific statements
            for i, line in enumerate(new_script_list):
                if (
                    line.count("Y column Name") > 0
                    or line.count("Y column Type")
                    or line.count("summary()")
                    or line.count("cv_results_") > 0
                    or line.count("sm.fit_sample(x,y)") > 0
                ):
                    line_indent = "" if len(line.lstrip()) == len(line) else line[0] * (len(line) - len(line.lstrip()))
                    new_script_list[i] = line_indent + "pass"

            new_script = "\n".join(new_script_list)

            if gen_mutation:
                with open(out_dir / mut_type / "script.py", "w", encoding="utf-8") as f:
                    f.write(new_script)

            muted_types.append(mut_type)

    if gen_df_dumper:
        t_train_stmt = list(train_stmt).copy()
        t_train_stmt[2] -= 1
        train_indent = (
            ""
            if len(script_list[t_train_stmt[2]].lstrip()) == len(script_list[t_train_stmt[2]])
            else script_list[t_train_stmt[2]][0]
            * (len(script_list[t_train_stmt[2]]) - len(script_list[t_train_stmt[2]].lstrip()))
        )
        new_script_list = script_list.copy()
        dataset_pkl_folder = out_dir / "df_dump"
        dataset_pkl_path = out_dir / "df_dump" / "df.pkl"
        to_inject = f"""
{train_indent}import pandas as pd
{train_indent}import scipy.sparse as sparse
{train_indent}import numpy as np
{train_indent}_____train = {train_X}
# {train_indent}if isinstance(_____train, sparse.spmatrix): {train_X} = pd.DataFrame(_____train.toarray())
{train_indent}if isinstance(_____train, np.ndarray): _____train = pd.DataFrame(_____train)
{train_indent}if isinstance(_____train, sparse.spmatrix): _____train = pd.DataFrame.sparse.from_spmatrix(_____train)
{train_indent}if isinstance(_____train, pd.Series): _____train = _____train.to_frame()
{train_indent}_____cols = ['feature_' + str(col) for col in _____train.columns]
{train_indent}_____train.columns = _____cols--
{train_indent}_____target = {train_Y}
{train_indent}if isinstance(_____target, pd.Series): _____target = _____target.to_frame()
{train_indent}if isinstance(_____target, np.ndarray): _____target = pd.DataFrame(_____target)
{train_indent}if isinstance(_____target, list): _____target = pd.DataFrame(_____target)
{train_indent}_____cols = ['target_' + str(col) for col in _____target.columns]
{train_indent}_____target.columns = _____cols
{train_indent}_____merge = _____train.merge(_____target, left_index=True, right_index=True)
# {train_indent}if len(_____merge) > 1000000: _____merge = _____merge.iloc[numpy.random.choice(_____merge.shape[0], 1000000, replace=False)]
# {train_indent}for col in _____merge: _____merge[col] = _____merge[col].sparse.to_dense() if isinstance(_____merge[col].array, pd.SparseArray) else _____merge[col]
{train_indent}import os
{train_indent}if not os.path.exists(f'{dataset_pkl_folder}'): os.makedirs(f'{dataset_pkl_folder}')
{train_indent}_____merge.to_pickle(f"{dataset_pkl_path}")
{train_indent}exit()
"""
        new_script_list.insert(t_train_stmt[2], to_inject)
        new_script = "\n".join(new_script_list)
        if not os.path.exists(out_dir / "df_dump"):
            os.makedirs(out_dir / "df_dump")
        with open(out_dir / "df_dump" / "script.py", "w", encoding="utf-8") as f:
            f.write(new_script)

    return m_type, muted_types
