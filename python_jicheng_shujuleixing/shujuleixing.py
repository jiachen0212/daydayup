# coding=utf-8
# python小语法: 常见的数据类型

# 1. 
# 元组: 不可被修改, 可作为dict的key
a = (1,2,3)
b = dict()
b[a] = 1
print(b)


# 2.
# set集合, 无需且不重复, 不可用sort进行排序
set1 = set(['a', 'b'])
print(set1)
set1.discard('a') # 删除元素
print(set1)
set1.update('a')  # 增加元素
print(set1)
print('a' in set1)  # 查询元素


# 3. 魔法函数: https://zhuanlan.zhihu.com/p/344951719
# https://zhuanlan.zhihu.com/p/443845745  我接触到的一个常用应用就是: class中构造和初始化.
# __name__ == "__main__": 这个的作用是, 本脚本可自行run, 但是本脚本被import的时候, __name__下的代码不会被执行.




