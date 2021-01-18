
from collections import OrderedDict
import pandas as pd
from comment_parser import comment_parser
import xlrd
import os
import subprocess
from tqdm import tqdm
from fnmatch import fnmatch
from timeout import timeout
from func_name_extract import *
import pandas as pd
import re


SpecialLine = {"**Parameters**":1, "**Description**":2, "**Context**":3,"**Return**":4}

def extract_segment(start,end,Lines):

    dict=OrderedDict()
    text={}
    beign = start
    while start !=end:
        if "**Parameters**" in Lines[start]:
            dict["Parameters"] = start
        if "**Description**" in Lines[start]:
            dict["Description"] = start
        if "**Context**" in Lines[start]:
            dict["Context"] = start
        if "**Return**" in Lines[start]:
            dict["Return"] = start
        start+=1

    key_list=[k for k in dict.keys()]
    value_list= [v for v in dict.values()]

    for i in range(len(key_list)-1):
        key = key_list[i]
        value = value_list[i]
        next_value = value_list[i+1]
        c = Lines[value+1:next_value]

        if key != "Parameters":
            text[key] = list_to_paragraph(c)
        else:
            text[key] = c
        print(key)
        print(c)
        print("===================== \n")


    start_index = value_list[0]

    text["abstract"] = list_to_paragraph(Lines[beign+1:start_index])
    print("abstract")
    print(text["abstract"] )
    print("===================== \n")

    end_parameter = key_list[-1]
    end_index = value_list[-1]
    if end_parameter =="Parameters":
        text[end_parameter] = Lines[end_index + 1:end]
    else:
        text[end_parameter] = list_to_paragraph(Lines[end_index + 1:end])

    print(end_parameter)
    print(text[end_parameter])
    print("===================== \n")
    return text


def list_to_paragraph(content):
    str=""
    for c in content:
        if c.strip() ==0:
            str += "\n"
            continue
        str += c.strip() + " "
    return str


def Get_script_comments(raw_data, tofile):
    fr = open(raw_data, 'r')
    Lines = fr.readlines()
    fr.close()
    i = 0
    func_list=[]
    abstract_list=[]
    desciption_list=[]
    parameter_list=[]
    context_list=[]
    return_list=[]


    while(i < len(Lines)):
        line = Lines[i]
        if fnmatch(line,".. c:function::*"):
            f_name = line.split("(")[0].split()[-1]
            start = i
            j = i+1
            while(j < len(Lines) and (not fnmatch(Lines[j],".. c:function::*") and not fnmatch(Lines[j],".. c:type::*"))):
                j = j+1
            end = j-1

            text=extract_segment(start,end,Lines)
            f_abtract=""
            f_parameter=""
            f_description=""
            f_context=""
            f_return=""
            if "abstract" in text.keys():
                f_abtract=text["abstract"]

            if "Parameters" in text.keys():
                f_parameter=text["Parameters"]

            if "Description" in text.keys():
                f_description=text["Description"]

            if "Context" in text.keys():
                f_context=text["Context"]

            if "Return" in text.keys():
                f_return=text["Return"]

            func_list.append(f_name)
            abstract_list.append(f_abtract)
            desciption_list.append(f_description)
            parameter_list.append(f_parameter)
            context_list.append(f_context)
            return_list.append(f_return)
        i+=1

    data={}
    data["func_list"]=func_list
    data["abstract_list"] = abstract_list
    data["desciption_list"] = desciption_list
    data["parameter_list"] = parameter_list
    data["context_list"] = context_list
    data["return_list"] = return_list
    df = pd.DataFrame(data)
    df.to_excel(tofile, header=True)


def list_all_files(rootdir):
    _files=[]
    list=os.listdir(rootdir)
    for i in range(0,len(list)):
        path=os.path.join(rootdir,list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        else:
            _files.append(path)
    return _files

def GetDoc(version,raw_data):
    files = list_all_files(version)
    fw = open(raw_data, 'w',errors='surrogateescape')
    count=0
    for _file in tqdm(files):
        filename, file_extension = os.path.splitext(_file)
        if file_extension != ".c" and file_extension != ".h":
            continue
        cmd = version+"/scripts/kernel-doc "+_file
        DocLog = ""
        try:
            DocLog = subprocess.check_output(cmd,shell=True)
        except subprocess.CalledProcessError as e:
            continue

        if len(DocLog.decode("utf-8") .split("\n")) == 1:
            continue
        try:
            fw.write(DocLog.decode("utf-8", 'replace') )
        except:
            f_error=open("error.txt","a+")
            f_error.write(version + " :: " + _file + "\n")
            f_error.close()

    fw.close()


def extract_type_segment(start,end,Lines):
    print("start lines:--")
    print(Lines[start])
    dict=OrderedDict()
    text={}
    beign = start
    while start !=end:
        if "**Syntax**" in Lines[start]:
            dict["Syntax"] = start
        if "**Members**" in Lines[start]:
            dict["Members"] = start
        if "**Description**" in Lines[start]:
            dict["Description"] = start
        if "**Definition**" in Lines[start]:
            dict["Definition"] = start
        if "**Constants**" in Lines[start]:
            dict["Constants"] = start
        start+=1

    key_list=[k for k in dict.keys()]
    value_list= [v for v in dict.values()]

    for i in range(len(key_list)-1):
        key = key_list[i]
        value = value_list[i]
        next_value = value_list[i+1]
        c = Lines[value+1:next_value]

        if key != "Members":
            text[key] = list_to_paragraph(c)
        else:
            text[key] = c
        print(key)
        print(c)
        print("===================== \n")


    start_index = value_list[0]

    text["abstract"] = list_to_paragraph(Lines[beign+1:start_index])
    print("abstract")
    print(text["abstract"] )
    print("===================== \n")

    end_parameter = key_list[-1]
    end_index = value_list[-1]
    if end_parameter =="Members":
        text[end_parameter] = Lines[end_index + 1:end]
    else:
        text[end_parameter] = list_to_paragraph(Lines[end_index + 1:end])

    print(end_parameter)
    print(text[end_parameter])
    print("===================== \n")
    return text


def Get_type_categories(raw_data,to_type_file):
    fr = open(raw_data, 'r')
    Lines = fr.readlines()
    fr.close()
    i = 0
    func_list=[]
    abstract_list=[]
    desciption_list=[]
    member_list=[]
    definition_list=[]
    constants_list=[]


    while(i < len(Lines)):
        line = Lines[i]
        if fnmatch(line,".. c:type::*"):
            f_name = line.split("(")[0].split()[-1]
            start = i
            j = i+1
            while(j < len(Lines) and (not fnmatch(Lines[j],".. c:type::*") and not fnmatch(Lines[j],".. c:function::*"))):
                j = j+1
            end = j-1

            text=extract_type_segment(start,end,Lines)
            f_abtract=""
            f_parameter=""
            f_description=""
            f_context=""
            f_return=""
            if "abstract" in text.keys():
                f_abtract=text["abstract"]

            if "Members" in text.keys():
                f_parameter=text["Members"]

            if "Description" in text.keys():
                f_description=text["Description"]

            if "Definition" in text.keys():
                f_context=text["Definition"]

            if "Constants" in text.keys():
                f_return=text["Constants"]

            func_list.append(f_name)
            abstract_list.append(f_abtract)
            desciption_list.append(f_description)
            member_list.append(f_parameter)
            definition_list.append(f_context)
            constants_list.append(f_return)
        i+=1

    data={}
    data["func_list"]=func_list
    data["abstract_list"] = abstract_list
    data["desciption_list"] = desciption_list
    data["member_list"] = member_list
    data["definition_list"] = definition_list
    data["constants_list"] = constants_list
    df = pd.DataFrame(data)
    df.to_excel(to_type_file, header=True)






def getMap():
    f=open("../coccinelle/func_list.txt","r")
    map={}
    for c in f.readlines():
        api = c.split(" :: ")[0]
        filename = c.split(" :: ")[1]
        map[api] = filename
    f.close()
    return map


@timeout(5)
def getComments(_file):
    comments = comment_parser.extract_comments(_file, mime='text/x-c')
    return comments

def containEnglish(str0):
    return bool(re.search('[a-zA-Z]',str0))

##get all regex comments
def Get_regex_Doc(count,version,all_func_file):
    files = list_all_files(version)
    #fw = open("data/linux_func_all_comments_5.8.txt", 'w')

    map=getMap()
    func_list=[]
    filename_list=[]
    comment_list=[]

    #fw = open("data/debug.txt", 'w')
    for _file in tqdm(files):
        filename, file_extension = os.path.splitext(_file)

        if file_extension != ".c" and file_extension != ".h":
            continue

        ##DEBUG
        '''
        if _file!="/home/yuexiao/Documents/research/diffcvss/extract_API_description/linux_kernel_mainline/linux/drivers/gpu/drm/amd/display/modules/color/color_gamma.c":
            continue
        '''

        try:
            comments= getComments(_file)
        except:
            print(_file)
            continue

        f=open(_file,"r")
        content=f.readlines()
        f.close()
        for _comment in comments:
            ##rule1: if not contain character
            if not containEnglish(_comment.text()):
                continue

            ## rule2: if the commets line, struct csio_rnode *rnode /*src/desctionation rnode*/ ignore this
            print(_file)
            print(_comment.line_number())
            current_line=content[_comment.line_number()-1]
            print(current_line)
            if "/*" in current_line and len(current_line.split("/*")[0].strip()) >0:
                continue
            if "//" in current_line and len(current_line.split("//")[0].strip()) >0:
                continue
            lines_count = _comment.text().split("\n")
            match_area=content[_comment.line_number()+len(lines_count)-1:_comment.line_number()+len(lines_count)+1]
            for index, code in enumerate(match_area):
                func=is_func(code)
                if func and len(func) >5:

                    count += 1
                    #fw.write("============" + _file + "===================" + '\n')
                    #fw.write(_comment.text()+ '\n')
                    #fw.write("============" + func + "===================" + '\n')
                    #fw.write(func + '\n')
                    func_list.append(func)
                    if func in map.keys():
                        filename_list.append(map[func])
                    else:
                        filename_list.append("")
                    comment_list.append(_comment.text())
                    #print(func)
                    #print(_comment.text())
                    #print("\n")
                    #fw.write("\n")
                    #fw.write("\n")
                    continue

                ###match corner case
                if index + 1 >= len(match_area)+1 or _comment.line_number()+len(lines_count)+index >= len(content):
                    continue

                conor_func = cornerCase(file_extension,code, content[_comment.line_number()+len(lines_count)-1 +index +1])
                if conor_func != None:
                    count += 1
                    #fw.write(conor_func + '\n')

                    func_list.append(conor_func)
                    if conor_func in map.keys():
                        filename_list.append(map[conor_func])
                    else:
                        filename_list.append("")
                    comment_list.append(_comment.text())


    #fw.close()


    data={}
    data["func_list"] = func_list
    data["filename_list"] = filename_list
    data["comment_list"] = comment_list
    df= pd.DataFrame(data)
    df.to_excel(all_func_file, header=True)
    print(count)
    return count





## GET func used regex
def get_regex(all_func,script_func,script_type,regex_to_file):
    script_api_sheet = pd.read_excel(script_func)
    script_type_sheet = pd.read_excel(script_type)

    script_api = script_api_sheet["func_list"].tolist()
    script_type= script_type_sheet["func_list"].tolist()

    regex_sheet = pd.read_excel(all_func)
    regx_func = regex_sheet["func_list"].tolist()
    comment_list = regex_sheet["comment_list"].tolist()

    new_regx_func = []
    new_comment_list = []
    count=0
    for index, func in enumerate(regx_func):
        if func in script_type or func in script_api:
            count+=1
            print(count)
            continue
        new_regx_func.append(func)
        new_comment_list.append(comment_list[index])

    data={}
    data["new_regx_func"] =new_regx_func
    data["new_comment_list"] = new_comment_list
    df = pd.DataFrame(data)
    #df.to_excel(regex_to_file, header=True)

    pre_process_regex_func(df, regex_to_file)



def pre_process_regex_func(df,tofile):
    regx_func = df["new_regx_func"].tolist()
    comment_list = df["new_comment_list"].tolist()

    processed_comment = []
    paramter_list =[]
    return_list=[]

    for index, func in enumerate(regx_func):
        comment = comment_list[index]
        paragraph, parameter, return_value = process_comment(comment)
        processed_comment.append(paragraph)
        paramter_list.append(parameter)
        return_list.append(return_value)

    data={}
    data["regx_func"] =regx_func
    data["processed_comment"] = processed_comment
    data["paramter_list"] = paramter_list
    data["return_list"] = return_list
    df = pd.DataFrame(data)
    df.to_excel(tofile, header=True)

##input regex comment, return paragraph, parameter, return
def process_comment(comment):
    paragraph=""
    parameter =""
    return_value=""
    index = 0
    comment_list = str(comment).split("\n")
    while index < len(comment_list):
        c=comment_list[index]
        c = c.replace("*"," ").strip()
        if len(c) ==0:
            c = "\n"
        if c.startswith("@"):
            parameter += c +"\n"
            index +=1
        elif c.lower().startswith("return"):
            return_value += c
            index+=1
            while index <  len(comment_list):
                return_value +=comment_list[index]
                index +=1
        else:
            paragraph += c + " "
            index +=1
    return paragraph, parameter, return_value

def get_regex_comment(version, topath):
    all_func_file =topath + "all_func_regex.xlsx"
    script_func =topath + "scripts_func.xlsx"
    script_type =topath + "scripts_type.xlsx"
    regex_to_file = topath + "regex_func.xlsx"
    if not os.path.exists(all_func_file):
        Get_regex_Doc(0, version, all_func_file)
    get_regex(all_func_file,script_func,script_type,regex_to_file)




def get_script_comment(version_path,topath):
    if not os.path.exists(topath):
        os.makedirs(topath)

    raw_script_file=topath + "raw_script_data.txt"
    if not os.path.exists(raw_script_file):
        GetDoc(version_path,raw_script_file)
    tofile = topath + "scripts_func.xlsx"
    to_type_file =topath + "scripts_type.xlsx"
    Get_script_comments(raw_script_file, tofile)
    Get_type_categories(raw_script_file, to_type_file)


def extract_comments_based_on_version( version_path,topath):
    #version_path = "/home/yuexiao/Documents/research/diffcvss/extract_API_description/andorid_kernel_source/kernel_common-android-mainline"
    #version_path = "/home/yuexiao/Documents/research/diffcvss/extract_API_description/linux_kernel_mainline/linux"

    ##get comments by built-in script
    get_script_comment(version_path,topath)

    ##get comments by regex expression
    get_regex_comment(version_path,topath)




if __name__=='__main__':
    version="linux-5.8"
    version_path="../../source_code/Linux/"+version
    topath="../../data/Linux/" + version+"/"
    flag="linux"
    extract_comments_based_on_version(version_path,topath)
