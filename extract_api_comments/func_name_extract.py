import os
import os.path
import pickle
import re
import sys
type_list = ["depot_stack_handle_t","inline",'static','int', 'char', 'float', 'double', 'bool', 'void', 'short', 'long', 'signed', 'struct',"unsigned","u32","irqreturn_t","const","size_t","#define","s32","enum","blk_status_t","u64","ssize_t","#define","extern","typedef"]


def pickle_dump(root_path, data, file_name):
    os.chdir(root_path)
    fp = open(file_name, "w")
    pickle.dump(data, fp)
    fp.close()

def pickle_load(root_path, file_name):
    os.chdir(root_path)
    fp_case = open(file_name, "r")
    dict_case = pickle.load(fp_case)
    fp_case.close()
    return dict_case


def is_valid_name(name):
    if re.match("[a-zA-Z_][a-zA-Z0-9_]*", name) == None:
        return False
    return True

def is_func(line):
#int, __int64, void, char*, char *, struct Node, long long int, (void *)
#int func(int a, int *b, (char *) c)
    line = line.strip()
    if len(line) < 2:
        return None
# Rule 1: assume the function name line must ends with ) or {;
    if line[-1] != ')' and line[-1] != '{' and  line[-1] != ','and  line[-1] != '(':
        return None
# Rule 2: (*) must in
    if '(' not in line :
        return None
# Rule 3: # stands for #include or other primitives; / start a comment
    if line[0] == '#' or line[0] == '/':
        return None

# replace pointer * and & as space
    line = re.sub('\*', ' ', line)
    line = re.sub('\&', ' ', line)


# replace '(' as ' ('
    #re.sub('(', ' ( ', line)
    line = re.sub('\(', ' \( ', line)
    line_split = line.split()

    if len(line_split) < 2:
        return None

    bracket_num = 0
    for ch in line:
        if ch == '(':
            bracket_num += 1

    has_type = False
    for type_a in type_list:
        if type_a == line_split[0]:
            has_type = True
    if has_type == False:
        return None

    if bracket_num == 1:
        for index in range(len(line_split)):
            if '(' in line_split[index]:
                return line_split[index - 1]
    else:
        line = re.sub('\(', ' ', line)
        line = re.sub('\)', ' ', line)
        line_split = line.split()
        index = 0
        for one in line_split:
            if is_valid_name(one):
                index += 1
                if index == 2:
                    return one
        return None



def dealwithHeader(line):
    if "=" in line:
        return None
    line = line.strip()
    if len(line) < 2:
        return None
    # Rule 1: assume the function name line must ends with ) or {;
    if line[-1] != ')' and line[-1] != '{' and line[-1] != ',' and line[-1] != '(' and  line[-1] != ';' :
        return None
    # Rule 2: (*) must in
    if '(' not in line:
        return None
    # Rule 3: # stands for #include or other primitives; / start a comment
    if line[0] == '#' or line[0] == '/':
        return None

    # replace pointer * and & as space
    line = re.sub('\*', ' ', line)
    line = re.sub('\&', ' ', line)

    # replace '(' as ' ('
    # re.sub('(', ' ( ', line)
    line = re.sub('\(', ' \( ', line)
    line_split = line.split()

    if len(line_split) < 2:
        return None

    bracket_num = 0
    for ch in line:
        if ch == '(':
            bracket_num += 1

    has_type = False
    for type_a in type_list:
        if type_a == line_split[0]:
            has_type = True
    if has_type == False:
        return None

    if bracket_num == 1:
        for index in range(len(line_split)):
            if '(' in line_split[index]:
                return line_split[index - 1]
    else:
        line = re.sub('\(', ' ', line)
        line = re.sub('\)', ' ', line)
        line_split = line.split()
        index = 0
        for one in line_split:
            if is_valid_name(one):
                index += 1
                if index == 2:
                    return one
        return None


def dealwithMultiLine(line,nextline):
    # replace pointer * and & as space
    line = re.sub('\*', ' ', line)
    line = re.sub('\&', ' ', line)

    # replace '(' as ' ('
    # re.sub('(', ' ( ', line)
    line = re.sub('\(', ' \( ', line)
    line_split = line.split()
    ##rule 1 :if "="
    if "=" in line or  "=" in nextline :
        return None
    ##rule 2 :if len(line_split) <1 or len(line_split) >3:
    if len(line_split) < 1 or len(line_split) >5:
        return None
    if line_split[0] not in type_list and "(" not in nextline:
        return None
    new_line = line+" " + nextline
    func = is_func(new_line)
    if func:
        return func

def cornerCase(file_extension, line, nextline):
    ## corner case 1
    func = dealwithMultiLine(line,nextline)
    if func and len(func.strip()) >=4:
        return func

    ## corner case 2 (hearder file)
    if file_extension==".h":
        func=dealwithHeader(line)
        if func in type_list:
            return None
        if func and len(func.strip()) >=4:
            f=open("header_add_func.txt","a+")
            #f.write(func+"\n")
            f.close()
            #print(func)
        return func




def func_name_extract(file_path):
    if not os.path.isfile(file_path):
        return

    fp = open(file_path, "r")
    fw=open("new_rules_func.txt","a+")
    func_list = []
    content = fp.readlines()
    for index, line in enumerate(content):
        if "iosf_mbi_unregister_pmic_bus_access_notifier(" not in line:
            continue
        func_name = is_func(line)
        if func_name != None:
            func_list.append(func_name)
            continue
        ## add new rules
        if index+1>=len(content):
            continue
        #if index !=251:
            #continue
        conor_func=cornerCase(line,content[index+1])
        if conor_func != None:
            func_list.append(conor_func)
            fw.write(conor_func+"\n")
    fp.close()
    fw.close()

    return func_list

def write_to_file(func_list, output_file):
    fp = open(output_file, "w")
    for one in func_list:
        fp.write(one + "\n")
    fp.close()

if __name__ == '__main__':
    file_path ="/home/yuexiao/Documents/research/diffcvss/extract_API_description/linux_kernel_mainline/linux/arch/x86/include/asm/iosf_mbi.h"
    func_list = func_name_extract(file_path)
    write_to_file(func_list, "func.txt")