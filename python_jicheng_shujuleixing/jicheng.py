# coding=utf-8
# 继承: 子继承父的部分属性和方法
# https://zhuanlan.zhihu.com/p/30239694  [单继承]



###################################################################
# 1. 继承

# 父类: 动物
class Animal(object):  #  python3中所有类都可以继承于object基类

    # 两个属性: name, age 
   def __init__(self, name, age):
       self.name = name
       self.age = age

    # 一个call方法
   def call(self):
       print(self.name, '会叫')


# 以下是子类: 
# 定义cat类, 继承animal父类
class Cat(Animal):
   def __init__(self,name,age,sex):
       # super初始化父类.
       super(Cat, self).__init__(name,age)  # 从Animal父类中继承想要的属性. [可部分继承.]
       # cat类自己再重新定义一个sex属性.
       self.sex=sex

# test1.
# if __name__ == '__main__':   
#    c = Cat('喵喵', 2, '男')   
#    c.call()   
###################################################################


###################################################################
# 2. 重构 和 多态 和 动态语言
    # 1. 子类重构父类的方法: 改写function
    # 2. 子类会优先执行自己改写的方法, 找不到方法的话才会去父类中找.  这就是, 多态. 
    # 3. 动态语言: 动态语言调用实例方法,不检查类型. 只要方法存在+参数正确, 就可以调用.
###################################################################

