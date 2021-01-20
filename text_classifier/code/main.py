
from text_classifier.code import av
from text_classifier.code import ac
from text_classifier.code import pr
from text_classifier.code import ui
from text_classifier.code import build
import os



def process_one_version(regex_func,script_func,total_file_path):
    #build AV
    if not os.path.exists(total_file_path):
        os.makedirs(total_file_path)
    av_total_file = total_file_path + "av_total.csv"
    if not os.path.exists(av_total_file):
        build.getAll_AV(regex_func, script_func, av_total_file)
    av_total_results = total_file_path+"av_result.xlsx"
    av.testall(av_total_file,av_total_results)


    #build AC
    if not os.path.exists(total_file_path):
        os.makedirs(total_file_path)
    ac_total_file = total_file_path + "ac_total.csv"
    if not os.path.exists(ac_total_file):
        build.getAll_AC(regex_func, script_func, ac_total_file)
    ac_total_results = total_file_path+"ac_result.xlsx"
    ac.testall(ac_total_file,ac_total_results)


    #build UI
    if not os.path.exists(total_file_path):
        os.makedirs(total_file_path)
    ui_total_file = total_file_path + "ui_total.csv"
    cmd="cp " + ac_total_file + " "+ ui_total_file
    os.system(cmd)
    ui_total_results = total_file_path+"ui_result.xlsx"
    ui.testall(ui_total_file,ui_total_results)


    #build PR
    if not os.path.exists(total_file_path):
        os.makedirs(total_file_path)
    pr_total_file = total_file_path + "pr_total.csv"
    if not os.path.exists(pr_total_file):
        build.getAll_PR(regex_func, script_func, pr_total_file)
    pr_total_results = total_file_path+"pr_result.xlsx"
    pr.testall(pr_total_file,pr_total_results)



if __name__ == '__main__':
    main_path="/Users/huthvincent/Documents/research/diffCVSS/pulish/data/Android/"
    api_main_path = "/Users/huthvincent/Documents/research/diffCVSS/pulish/API/Android/"
    versions = os.listdir(main_path)
    for version in versions:
        regex_func=main_path + version+ "/regex_func.xlsx"
        script_func=main_path + version+ "/scripts_func.xlsx"
        total_file_path=api_main_path + version + "/"
        process_one_version(regex_func, script_func, total_file_path)




