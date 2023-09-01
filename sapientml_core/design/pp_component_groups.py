"""
To design the search space of SapientML, we group semantically similar low-level APIs into a high level group and assign
a group label. For example pandas.fillna and sklearn.SimpleImputer both are used to fill out missing values in the dataframe.

This semantic grouping helps SapientML keep the search space simple and discover more accurate relationships between
various ML components and meta-features of the dataset during training of skeleton predictor.
"""


drop_label_list = [
    "PREPROCESS:MissingValues:dropna:pandas",
    "PREPROCESS:MissingValues:notnull:pandas",
    "PREPROCESS:MissingValues:isnull:pandas",
]
filler_label = [
    "PREPROCESS:MissingValues:fillna:pandas",
    "PREPROCESS:MissingValues:SimpleImputer:sklearn",
    "PREPROCESS:MissingValues:KNNImputer:sklearn",
    "PREPROCESS:MissingValues:replace:pandas",
    "PREPROCESS:MissingValues:random:custom",
    "PREPROCESS:MissingValues:interpolate:sklearn",
]
in_place_converter = [
    "PREPROCESS:Category:LabelEncoder:sklearn",
    "PREPROCESS:Category:factorize:pandas",
    "PREPROCESS:Category:replace:pandas",
    "PREPROCESS:Category:map:custom",
    "PREPROCESS:Category:apply:pandas",
    "PREPROCESS:Category:custom:pandas",
]
one_hot = [
    "PREPROCESS:Category:get_dummies:pandas",
    "PREPROCESS:Category:OneHotEncoder:sklearn",
    "PREPROCESS:Category:LabelBinarizer:sklearn",
]

text_vect = ["PREPROCESS:Text:CountVectorizer:sklearn", "PREPROCESS:Text:TfidfVectorizer:sklearn"]

scaling = [
    "PREPROCESS:Scaling:STANDARD:sklearn",
    "PREPROCESS:Scaling:MIN_MAX:custom",
    "PREPROCESS:Scaling:MIN_MAX:sklearn",
    "PREPROCESS:Scaling:STANDARD:custom",
    "PREPROCESS:Scaling:Robust:sklearn",
    "PREPROCESS:Scaling:STANDARD:Pandas",
    "PREPROCESS:Scaling:normalize:sklearn",
    "PREPROCESS:Scaling:normalize:Pandas",
    "PREPROCESS:Scaling:STANDARD:pandas",
]

date = [
    "PREPROCESS:GenerateColumn:date:pandas",
    "PREPROCESS:GenerateColumn:DATE:pandas",
    "PREPROCESS:GenerateColumn:DATE:custom",
]

text_processing = [
    "PREPROCESS:Text:lower:pandas",
    "PREPROCESS:Text:remove_non_alpha:custom",
    "PREPROCESS:Text:tokenize:nltk",
    "PREPROCESS:Text:Lemmtize:nltk",
]

balancing = [
    "PREPROCESS:Balancing:SMOTE:imblearn",
    "PREPROCESS:Balancing:resample:custom",
    "PREPROCESS:Balancing:sample:custom",
]

log_transform = [
    "PREPROCESS:Scaling:log1p:numpy",
    "PREPROCESS:Scaling:power:custom",
    "PREPROCESS:Scaling:log:numpy",
    "PREPROCESS:Scaling:sqrt:numpy",
    "PREPROCESS:Scaling:exp:numpy",
    "PREPROCESS:Scaling:log:custom",
    "PREPROCESS:Scaling:power_transform:sklearn",
]
