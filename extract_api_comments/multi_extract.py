from multiprocessing import Pool
import time, os, random
import sys
from extract_api_comments import  getDescriptions

def worker(flag, version):
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

    po = Pool(10)
    versions = os.listdir(path)
    for i in range(len(versions)):
        # Pool().apply_async(要调用的目标,(传递给目标的参数元祖,))
        # 每次循环将会用空闲出来的子进程去调用目标
        po.apply_async(worker, (flag,versions[i],))

    print("----start----")
    po.close()  # 关闭进程池，关闭后po不再接收新的请求
    po.join()  # 等待po中所有子进程执行完成，必须放在close语句之后
    print("-----end-----")


if __name__ == '__main__':
    main()