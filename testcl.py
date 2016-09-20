#!/bin/bash python
# -*- coding:utf8 -*-

import pandas as pd
import numpy as np
import pdb
import sys


# no comma
def nocomma_pivot(df, prefix, index, column, value):
    df = df.pivot(index, column, value)
    col = [c for c in df.columns if c != ""]
    df = df[col]
    df.columns = [prefix + c for c in df.columns]
    df = df.reset_index(level=0).fillna(0)
    df.replace("Fr", 0, inplace=True)
    df.replace("T", 0, inplace=True)
    return df


# comma
def comma_pivot(df, index, column, value):
    result = pd.DataFrame(columns=[index, "name", "value"])
    lineno = 0
    for col in df.iterrows():
        split_name = col[1][column].split(",")
        split_value = col[1][value].split(",")
        for i in range(0, len(split_name)):
            result.loc[lineno] = [col[1][index], split_name[i], split_value[i]]
            lineno += 1
    result = result.drop_duplicates([index, "name"])
    result = result.pivot(index, "name", "value").reset_index(level=0)
    result = result[[c for c in result.columns if c != ""]].fillna(0)
    result.replace("", 0, inplace=True)
    return result


# color str to vaule
# true false to value
def str_to_int(df, cols):
    for name in cols:
        dict_color = df.loc[:, name].unique()
        dict_color = [i for i in dict_color if i != '' and i != 'None']
        dict_color.sort()
        df.replace('', 0, inplace=True)
        df.replace('None', 0, inplace=True)
        for i in range(1, len(dict_color) + 1):
            df.replace(dict_color[i - 1], i, inplace=True)

# va_color_dominant_colors
def dominant(df,index, cols):
    result = pd.DataFrame(columns=[index, cols])
    lineno = 0
    for col in df.iterrows():
        split_name = col[1][cols].split(",")
        for i in range(0, len(split_name)):
            result.loc[lineno] = [col[1][index], split_name[i]]
            lineno += 1
    dict_color = result.loc[:, cols].unique()
    dict_color = [i for i in dict_color if i != '' and i != 'None']
    dict_color.sort()
    result.replace('', 0, inplace=True)
    result.replace('None', 0, inplace=True)
    for i in range(1, len(dict_color) + 1):
        result.replace(dict_color[i - 1], i, inplace=True)
    return result

# "va_faces_age"
def faceage(df,index,cols):
    result = pd.DataFrame(columns=[index, cols])
    lineno = 0
    for col in df.iterrows():
        split_name = col[1][cols].split(",")
        for i in range(0, len(split_name)):
            result.loc[lineno] = [col[1][index], split_name[i]]
            lineno += 1
    result[cols] = result[cols].apply(pd.to_numeric)
    newresult=result.groupby(index).mean().reset_index(level=0)
    return newresult

# "va_faces_gender"
def genderpivot(df,index,cols):
    result = pd.DataFrame(columns=[index, cols])
    lineno = 0
    for col in df.iterrows():
        split_name = col[1][cols].split(",")
        for i in range(0, len(split_name)):
            result.loc[lineno] = [col[1][index], split_name[i]]
            lineno += 1
    result.loc[:, "value"] = 1
    newresult = pd.pivot_table(result, columns = cols, values = "value", index = "pic_id", aggfunc = "sum").reset_index(level=0)
    newresult.replace("NaN", 0, inplace=True)
    return newresult

'''
# "va_faces_rectangle"
def rectangle(df,index,cols):
    result = pd.DataFrame(columns=[index, cols])
    lineno = 0
    for col in df.iterrows():
        split_name = col[1][cols].split(":")
        for i in range(0, len(split_name)):
            result.loc[lineno] = [col[1][index], split_name[i]]
            lineno += 1
    result = result.join(result.va_faces_rectangle.apply(lambda x: pd.Series(x.split(',')))).fillna("0")
    result.columns = [index,cols,"newx","newy","neww","newh"]
    changecols=["newx","newy","neww","newh"]
    result[changecols] = result[changecols].apply(pd.to_numeric)
    data = newdf.loc[:,["pic_id","width","height"]]
    newresult = pd.merge(result,data,on="pic_id")
    newresult.loc[:, "bais"] = abs(newresult.loc[:, "width"] / 2 - newresult.loc[:, "newx"]) + abs(newresult.loc[:, "height"] / 2 - newresult.loc[:, "newy"])
    newresult.loc[:, "ratio"] = (newresult.loc[:, "neww"] * newresult.loc[:, "newh"]) / (newresult.loc[:, "width"] * newresult.loc[:, "height"])
    lastresult = newresult.loc[:, ["pic_id", "bais", "ratio"]]
    lastresult = lastresult.groupby("pic_id").mean()
    return lastresult
'''
# process
def process_table_feature(df):
    df.columns = ["pid", "uid", "pic_id", "image_id", "lat", "lon", "places_labels", "places_weight", "timestamp",
                  "width", "height", "va_adult_is_adult_content", "va_adult_is_racy_content", "va_adult_racy_score",
                  "va_adult_score", "va_color_accent_color", "va_color_dominant_background",
                  "va_color_dominant_foreground", "va_color_is_bw_img", "va_description_captions_confidences",
                  "va_description_captions_texts", "va_description_tags", "va_faces_count", "va_image_type_clip_art",
                  "va_image_type_line_drawing", "va_categories_names", "va_categories_scores", "va_tags_confidences",
                  "va_tags_names", "va_color_dominant_colors", "va_faces_age", "va_faces_gender", "va_faces_rectangle"]
    newdf = df.loc[:, ["pic_id", "lat", "lon", "places_labels", "places_weight", "timestamp", "width", "height",
                       "va_adult_is_adult_content", "va_adult_is_racy_content", "va_adult_racy_score", "va_adult_score",
                       "va_color_accent_color", "va_color_dominant_background", "va_color_dominant_foreground",
                       "va_color_is_bw_img", "va_description_captions_confidences", "va_description_captions_texts",
                       "va_description_tags", "va_faces_count", "va_image_type_clip_art", "va_image_type_line_drawing",
                       "va_categories_names", "va_categories_scores", "va_tags_confidences", "va_tags_names",
                       "va_color_dominant_colors", "va_faces_age", "va_faces_gender", "va_faces_rectangle"]]
    tonum_cols = ["lat", "lon", "width", "height", "timestamp", "va_description_captions_confidences",
                  "va_image_type_clip_art", "va_image_type_line_drawing", "va_faces_count"]
    newdf[tonum_cols] = newdf[tonum_cols].apply(pd.to_numeric)
    # color 16 to 10
    newdf[["va_color_accent_color"]] = newdf[["va_color_accent_color"]].replace("", "0")
    newdf[["va_color_accent_color"]] = newdf[["va_color_accent_color"]].fillna("0")
    newdf[["va_color_accent_color"]] = newdf["va_color_accent_color"].apply(lambda x: int(x, 16))
    allnum = newdf.loc[:,
             ["pic_id", "lat", "lon", "width", "height", "timestamp", "va_description_captions_confidences",
              "va_image_type_clip_art", "va_image_type_line_drawing", "va_faces_count", "va_color_accent_color"]]
    # no comma pivot
    adult = newdf.loc[:, ["pic_id", "va_adult_is_adult_content", "va_adult_score"]].fillna("").replace(False,
                                                                                                       "Fr").replace(
        True, "T")
    adult = nocomma_pivot(adult, 'adult_', "pic_id", "va_adult_is_adult_content", "va_adult_score")
    racy = newdf.loc[:, ["pic_id", "va_adult_is_racy_content", "va_adult_racy_score"]].fillna("").replace(False,
                                                                                                          "Fr").replace(
        True, "T")
    racy = nocomma_pivot(racy, 'racy_', "pic_id", "va_adult_is_racy_content", "va_adult_racy_score")
    # comma pivot
    #places = newdf.loc[:, ["pic_id", "places_labels", "places_weight"]]
    categories = newdf.loc[:, ["pic_id", "va_categories_names", "va_categories_scores"]].fillna("")
    tags = newdf.loc[:, ["pic_id", "va_tags_names", "va_tags_confidences"]].fillna("")
    #places = comma_pivot(places, "pic_id", "places_labels", "places_weight")
    categories = comma_pivot(categories, "pic_id", "va_categories_names", "va_categories_scores")
    tags = comma_pivot(tags, "pic_id", "va_tags_names", "va_tags_confidences")
    # replace str to int
    replace_cols = ["va_color_dominant_background", "va_color_dominant_foreground", "va_color_is_bw_img"]
    replacedf = newdf.loc[:,["pic_id", "va_color_dominant_background", "va_color_dominant_foreground", "va_color_is_bw_img"]]
    str_to_int(replacedf, replace_cols)
    #va_color_dominant_colors
    dominantcolor = df.loc[:, ["pic_id", "va_color_dominant_colors"]].fillna("0")
    dominantcolor = dominant(dominantcolor, "pic_id", "va_color_dominant_colors")
    #va_faces_age
    agedata = newdf.loc[:, ["pic_id", "va_faces_age"]].fillna("0")
    agedata = faceage(agedata, "pic_id", "va_faces_age")
    #va_faces_gender
    genderdata = newdf.loc[:, ["pic_id", "va_faces_gender"]].fillna("0")
    genderdata = genderpivot(genderdata, "pic_id", "va_faces_gender")

    #va_faces_rectangle
    #rectangledata = newdf.loc[:, ["pic_id", "va_faces_rectangle"]].fillna("0")
    #rectangledata = rectangle(rectangledata, "pic_id", "va_faces_rectangle")
    # join
    # 0.981#resultdf=allnum.merge(adult,on="pic_id").merge(racy,on="pic_id").merge(places,on="pic_id").merge(categories,on="pic_id").merge(tags,on="pic_id")#.merge(replacedf,on="pic_id")
    # 0.989#resultdf=allnum.merge(adult,on="pic_id")#.merge(racy,on="pic_id").merge(places,on="pic_id").merge(categories,on="pic_id").merge(tags,on="pic_id")#.merge(replacedf,on="pic_id")
    # 0.990#resultdf=allnum.merge(adult,on="pic_id")#.merge(racy,on="pic_id").merge(places,on="pic_id").merge(categories,on="pic_id").merge(tags,on="pic_id")#.merge(replacedf,on="pic_id")
    # 0.988#resultdf=allnum.merge(adult,on="pic_id").merge(racy,on="pic_id")#.merge(places,on="pic_id").merge(categories,on="pic_id").merge(tags,on="pic_id")#.merge(replacedf,on="pic_id")
    # 0.986#resultdf=allnum.merge(adult,on="pic_id").merge(racy,on="pic_id").merge(categories,on="pic_id")#.merge(places,on="pic_id").merge(categories,on="pic_id").merge(tags,on="pic_id")#.merge(replacedf,on="pic_id")
    # 0.986#resultdf=allnum.merge(adult,on="pic_id").merge(racy,on="pic_id").merge(categories,on="pic_id")#.merge(places,on="pic_id").merge(tags,on="pic_id")#.merge(replacedf,on="pic_id")
    # resultdf=allnum.merge(adult,on="pic_id").merge(racy,on="pic_id").merge(categories,on="pic_id").merge(tags,on="pic_id")#.merge(places,on="pic_id").merge(categories,on="pic_id")#.merge(replacedf,on="pic_id")
    # 0.992261#resultdf = allnum.merge(adult, on="pic_id").merge(racy, on="pic_id").merge(categories, on="pic_id").merge(tags,on="pic_id").merge(replacedf, on="pic_id").merge(dominantcolor, on="pic_id")
    #0.992668#resultdf = allnum.merge(adult, on="pic_id").merge(racy, on="pic_id").merge(categories, on="pic_id").merge(tags,on="pic_id").merge(replacedf, on="pic_id").merge(dominantcolor, on="pic_id").merge(agedata, on="pic_id")
    # 0.993483#resultdf = allnum.merge(adult, on="pic_id").merge(racy, on="pic_id").merge(categories, on="pic_id").merge(tags,on="pic_id").merge(replacedf, on="pic_id").merge(dominantcolor, on="pic_id").merge(agedata, on="pic_id").merge(genderdata, on="pic_id")
    #0.993483#resultdf = allnum.merge(adult, on="pic_id").merge(racy, on="pic_id").merge(categories, on="pic_id").merge(tags,on="pic_id").merge(replacedf, on="pic_id").merge(dominantcolor, on="pic_id").merge(agedata, on="pic_id").merge(genderdata,on="pic_id").merge(places,on="pic_id")
    resultdf = allnum.merge(adult, on="pic_id").merge(racy, on="pic_id").merge(categories, on="pic_id").merge(tags,on="pic_id").merge(replacedf, on="pic_id").merge(dominantcolor, on="pic_id").merge(agedata, on="pic_id").merge(genderdata,on="pic_id")
    return resultdf

# turn the data in a (samples, feature) matrix
positive = pd.read_csv("./positive.csv", sep="\t").iloc[:, 1:34]
negative = pd.read_csv("./negative.csv", sep="\t").iloc[:, 1:34]
positive = process_table_feature(positive)
positive.loc[:, "label"] = 1
negative = process_table_feature(negative)
negative.loc[:, "label"] = 2
train_data = positive.append(negative, ignore_index=True)
train_data = train_data[[c for c in train_data.columns if c != "pic_id"]].fillna(0).dropna()
train_data = train_data.apply(pd.to_numeric)
column_name=list(train_data)
column_name_dataframe = pd.DataFrame({"column_name":column_name})
#column_name_dataframe.to_csv()

'''
#Split the dataset in two  parts
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_data[[c for c in train_data.columns if c != "label"]],
                                                    train_data[["label"]], test_size=0.30, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)
print "train 准确率 %f" % (rf.score(X_train, y_train))
print "test 准确率 %f" % (rf.score(X_test, y_test))


# compute the feature importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
'''
#
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(train_data[[c for c in train_data.columns if c != "label"]],train_data[["label"]])
from sklearn.externals import joblib
joblib.dump(rf, 'cl.pkl')



