# coding=utf-8
# 面向对象编程

# 1. __name 前置双下划线, 仅可内部访问, 外部不可
# 2. __fun__ 前后均双下划线, 是类的内置方法. 

# __call__
class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self, *args, **kwargs):
        print("执行实例方法call方法")

p = Person("test", 26)
print(p())   # 因为类中有__call__, 故可这样调用类 



# __str__
class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"my name is {self.name} age is {self.age}"

p = Person("baozi", 26)
print(str(p))
print(p)    # 等价


# __repr__
class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"my name is {self.name} age is {self.age}"

    def __repr__(self):
        return "Person('%s', %s)" % (self.name, self.age)

p = Person("baozi", 26)
print(p)    # my name is baozi age is 26

print(repr(p))  # Person('baozi', 26)



# __getitem__()、__setitem__()、__delitem__() 类似字典的取值, 赋值, 删除 功能
class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __getitem__(self, key):
        print("通过 [] 取值时，调用了我")
        temp = '{}, {}'.format(self.name, self.age)
        return temp

p = Person("baozi", 26)
print(p["age"])    # 通过 [] 取值时，调用了我


# @staticmethod: 装饰为静态方法, 就不需要传入self参数也可正常访问. 
class Person(object):
    def __init__(self, name, age):
        self._name = name
        self._age = age

    def talk(self):
        print(f"name is {self._name} age is {self._age}")

    @staticmethod
    def static_talk(name, age): # 这里无需再传递self，函数不用再访问类
        print(f"name is {name} age is {age}")

p = Person("baozi", 26) 
print(p.talk())
print(p.static_talk('chenjia', 26))



# @classmethod: 装饰为类方法, 无需实例化, 就可访问.
class Person(object):
    def __init__(self, name, age):
        self._name = name
        self._age = age

    @classmethod
    def static_talk(cls, name, age):  # 注意第一个参数: cls
        print(f"my name is {name} age is {age}")

print(Person.static_talk("jia.chen", 26))





