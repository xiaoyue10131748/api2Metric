import random
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import xlrd
import pickle
import ast
from text_classifier.code.TextPreprocessing import PuritySegmens

f = open('../data/file_dir_map', 'rb')
file_dir_map=pickle.load(f)


def get_comments_based_on_API_pr(func,regex_sheet,script_sheet):
    func_list = script_sheet["func_list"].tolist()
    abstract_list  = script_sheet["abstract_list"].tolist()
    desciption_list  = script_sheet["desciption_list"].tolist()

    if func == "posix_acl_permission":
        print(func)
    if func.strip() in func_list:
        index = func_list.index(func.strip())
        abstract = str(abstract_list[index])
        if abstract =="nan":
            abstract=""

        description = str(desciption_list[index])
        if description == "nan":
            description=""


        complete_comment = str(func)+ str(abstract) + ". " + str(description)
        clean_comment = PuritySegmens(complete_comment)
        return clean_comment


    regex_func=regex_sheet["regx_func"].tolist()
    processed_comment =regex_sheet["processed_comment"].tolist()

    if func.strip() in regex_func:
        index = regex_func.index(func)
        return PuritySegmens(str(func) + str(processed_comment[index]))



def batch_process_pr(api_list,sheet_regex,sheet_script):
    comment_list=[]
    i=0
    for api in api_list:
        print(i)
        i+=1
        comm = get_comments_based_on_API_pr(str(api).strip(),sheet_regex,sheet_script)
        comment_list.append(str(comm))
    return comment_list




def batch_process_ui(api_list,script_sheet,regex_sheet):
    comment_list=[]
    i=0
    for api in api_list:
        #print(i)
        i+=1
        if i ==72:
            print(i)
        #print(api)
        comments=get_comments_based_on_API_ui(api,script_sheet,regex_sheet)
        try:
            comm = PuritySegmens(comments)
            comment_list.append(str(comm))
        except:
            comment_list.append("")
            print(comments)
    return comment_list



def batch_process_av(api_list,sheet_script,sheet_regex):
    comment_list=[]
    i=0
    for api in api_list:
        print(i)
        i+=1
        if i ==72:
            print(i)
        print(api)
        comm = PuritySegmens(get_comments_based_on_API_av(str(api).strip(),sheet_script,sheet_regex))
        comment_list.append(str(comm))


    return comment_list


def get_comments_based_on_API_av(func,script_sheet,regex_sheet):
    try:
        path = file_dir_map[func].replace("/", " ").replace(".c", " ").replace(".h", " ")
    except:
        path=""

    func_list = script_sheet["func_list"].tolist()
    abstract_list  = script_sheet["abstract_list"].tolist()
    desciption_list  = script_sheet["desciption_list"].tolist()
    parameter_list = script_sheet["parameter_list"].tolist()
    context_list = script_sheet["context_list"].tolist()
    return_list = script_sheet["return_list"].tolist()

    if func.strip() in func_list:
        index = func_list.index(func.strip())
        abstract = str(abstract_list[index])
        if abstract =="nan":
            abstract=""

        description = str(desciption_list[index])
        if description == "nan":
            description=""

        parameter = parameter_list[index]
        res = ast.literal_eval(parameter)
        p_list=""
        for  r in res:
            if len(r.strip())==0:
                continue
            else:
                p_list += str(r.strip().replace("``" , "")) +" "


        context = str(context_list[index])
        if context == "nan":
            context = ""

        return_v = str(return_list[index])
        if return_v =="nan":
            return_v = ""

        complete_comment = str(path) + str(abstract) + " " + str(description) + " " +str(p_list) + " " + str(context) + " " + str(return_v)
        return complete_comment


    regex_func=regex_sheet["regx_func"].tolist()
    processed_comment =regex_sheet["processed_comment"].tolist()
    paramter_list = regex_sheet["paramter_list"].tolist()
    return_list = regex_sheet["return_list"].tolist()
    if func.strip() in regex_func:
        index = regex_func.index(func)
        description = str(processed_comment[index]).replace("\n", " ")
        if description == "nan":
            description =""

        params = str(paramter_list[index]).replace("\n", " ")
        if params == "nan":
            params=""

        return_v =str(return_list[index]).replace("\n", " ")
        if return_v =="nan":
            return_v =""

        return str(path) + description + " " + params + " "+return_v


def get_comments_based_on_API_ui(func,script_sheet,regex_sheet):
    func_list = script_sheet["func_list"].tolist()
    abstract_list  = script_sheet["abstract_list"].tolist()
    desciption_list  = script_sheet["desciption_list"].tolist()
    parameter_list = script_sheet["parameter_list"].tolist()
    context_list = script_sheet["context_list"].tolist()
    return_list = script_sheet["return_list"].tolist()

    if func in func_list:
        index = func_list.index(func)
        abstract = str(abstract_list[index])
        if abstract =="nan":
            abstract=""

        description = str(desciption_list[index])
        if description == "nan":
            description=""

        parameter = parameter_list[index]
        res = ast.literal_eval(parameter)
        p_list=""
        for  r in res:
            if len(r.strip())==0:
                continue
            else:
                p_list += str(r.strip().replace("``" , "")) +" "


        context = str(context_list[index])
        if context == "nan":
            context = ""

        return_v = str(return_list[index])
        if return_v =="nan":
            return_v = ""

        complete_comment =str(func)+" "+ str(abstract) + " " + str(description) + " " +str(p_list) + " " + str(context) + " " + str(return_v)
        return complete_comment


    regex_func=regex_sheet["regx_func"].tolist()
    processed_comment =regex_sheet["processed_comment"].tolist()
    paramter_list = regex_sheet["paramter_list"].tolist()
    return_list = regex_sheet["return_list"].tolist()
    if func in regex_func:
        index = regex_func.index(func)
        description = str(processed_comment[index]).replace("\n", " ")
        if description == "nan":
            description =""

        params = str(paramter_list[index]).replace("\n", " ")
        if params == "nan":
            params=""

        return_v =str(return_list[index]).replace("\n", " ")
        if return_v =="nan":
            return_v =""

        return str(func) + " " +description + " " + params + " "+return_v




def getAll_AC(regex_func, script_func,total_file):
    sheet_script = pd.read_excel(regex_func)
    sheet_regex = pd.read_excel(script_func)
    func_1 = sheet_regex["regx_func"].tolist()
    X_1 = batch_process_ui(func_1,sheet_script,sheet_regex)
    tag_1 = [0] * len(func_1)

    func_2 = sheet_script["func_list"].tolist()
    X_2 = batch_process_ui(func_2,sheet_script,sheet_regex)
    tag_2 = [0] * len(func_2)

    x1_rows = zip(tag_1, func_1, X_1)
    with open(total_file, "w") as f:
        writer = csv.writer(f)
        for row in x1_rows:
            writer.writerow(row)

    x2_rows = zip(tag_2, func_2, X_2)
    with open(total_file, "a+") as f:
        writer = csv.writer(f)
        for row in x2_rows:
            writer.writerow(row)





def getAll_PR(regex_func, script_func,total_file):
    sheet_script = pd.read_excel(regex_func)
    sheet_regex = pd.read_excel(script_func)
    func_1 = sheet_regex["regx_func"].tolist()
    X_1 = batch_process_pr(func_1,sheet_script,sheet_regex)
    tag_1 = [0] * len(func_1)

    func_2 = sheet_script["func_list"].tolist()
    X_2 = batch_process_pr(func_2,sheet_script,sheet_regex)
    tag_2 = [0] * len(func_2)

    x1_rows = zip(tag_1, func_1, X_1)
    with open(total_file, "w") as f:
        writer = csv.writer(f)
        for row in x1_rows:
            writer.writerow(row)

    x2_rows = zip(tag_2, func_2, X_2)
    with open(total_file, "a+") as f:
        writer = csv.writer(f)
        for row in x2_rows:
            writer.writerow(row)





def getAll_AV(regex_func, script_func,total_file):

    sheet_script = pd.read_excel(script_func)
    sheet_regex = pd.read_excel(regex_func)
    func_1 = sheet_regex["regx_func"].tolist()
    X_1 = batch_process_av(func_1,sheet_script,sheet_regex)
    tag_1 = [0] * len(func_1)

    func_2 = sheet_script["func_list"].tolist()
    X_2 = batch_process_av(func_2,sheet_script,sheet_regex)
    tag_2 = [0] * len(func_2)


    x1_rows = zip(tag_1, func_1, X_1)
    with open(total_file, "w") as f:
        writer = csv.writer(f)
        for row in x1_rows:
            writer.writerow(row)

    x2_rows = zip(tag_2, func_2, X_2)
    with open(total_file, "a+") as f:
        writer = csv.writer(f)
        for row in x2_rows:
            writer.writerow(row)



def getAll_UI(regex_func, script_func,total_file):
    sheet_script = pd.read_excel(regex_func)
    sheet_regex = pd.read_excel(script_func)
    func_1 = sheet_regex["regx_func"].tolist()
    X_1 = batch_process_ui(func_1,sheet_script,sheet_regex)
    tag_1 = [0] * len(func_1)

    func_2 = sheet_script["func_list"].tolist()
    X_2 = batch_process_ui(func_2,sheet_script,sheet_regex)
    tag_2 = [0] * len(func_2)

    x1_rows = zip(tag_1, func_1, X_1)
    with open(total_file, "w") as f:
        writer = csv.writer(f)
        for row in x1_rows:
            writer.writerow(row)

    x2_rows = zip(tag_2, func_2, X_2)
    with open(total_file, "a+") as f:
        writer = csv.writer(f)
        for row in x2_rows:
            writer.writerow(row)



if __name__ == '__main__':

    #getAll("PR", "linux")
    #getAll("PR","android")
    #getAll("AC", "linux")
    #getAll("AC", "android")
    getAll_AV("linux")
    getAll_UI("linux")
    #second_time_label_pr_0()
