def genderpivot(df,index,cols):
    result = pd.DataFrame(columns=[index, cols])
    lineno = 0
    for col in df.iterrows():
        split_name = col[1][cols].split(",")
        for i in range(0, len(split_name)):
            result.loc[lineno] = [col[1][index], split_name[i]]
            lineno += 1
    result.loc[:, "value"] = 1
    newresult = pd.pivot_table(result, columns = cols, values = "value", index = "pic_id", aggfunc = "sum")
    newresult.replace("", 0, inplace=True)
    return newresult

genderdata = newdf.loc[:,["pic_id","va_faces_gender"]].fillna("0")
genderdata = genderpivot(genderdata,"pic_id","va_faces_gender")