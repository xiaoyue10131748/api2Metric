from multiprocessing import Pool
import time, os, random
import sys
import  getDescriptions

def exe(flag, version):
    t_start = time.time()
    print("%sbegin excuting %d" % (version, os.getpid()))

    if flag == "linux":
        version_path = "../../source_code/Linux/" + version
        topath = "../../data/Linux/" + version + "/"

        getDescriptions.extract_comments_based_on_version( version_path, topath)

    if flag == "android":
        version_path = "../../source_code/Android/" + version + "/kernel_common"
        topath = "../../data/Android/" + version + "/"
        getDescriptions.extract_comments_based_on_version(version_path, topath)

    time.sleep(random.random() * 2)
    t_stop = time.time()
    print(version, "excute end %0.2f" % (t_stop - t_start))


def main():
    flag = sys.argv[1]
    path = sys.argv[2]
    #flag = "linux"

    #path = "../../source_code/Linux/"

    #po = Pool(10)
    versions = os.listdir(path)
    print(versions)
    for i in range(len(versions)):
        print(versions[i])
        exe(flag,versions[i])

    print("----start----")
    #po.close()  
    #po.join()  
    print("-----end-----")


if __name__ == '__main__':
    main()
