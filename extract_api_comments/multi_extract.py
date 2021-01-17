from multiprocessing import Pool
import time, os, random
import sys
from extract_api_comments import  getDescriptions

def worker(flag, version):
    t_start = time.time()  # 获取当前系统时间，长整型，常用来测试程序执行时间
    print("%s开始执行,进程号为%d" % (version, os.getpid()))
    # random.random()随机生成0~1之间的浮点数
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
    print(version, "执行完毕，耗时%0.2f" % (t_stop - t_start))


def main():
    flag = sys.argv[1]
    path = sys.argv[2]
    #flag = "linux"

    #path = "../../source_code/Linux/"

    po = Pool(3)  # 定义一个进程池，最大进程数3，大小可以自己设置，也可写成processes=3
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