# coding=utf-8
# 面向对象编程

# 1. 
# __name 前置双下划线, 仅可内部访问, 外部不可
# __fun__ 前后均双下划线, 是类的内置方法. 
class Student:

    def __init__(self, name, score):
        self.name = name
        self.__score = score

    # 定义打印学生信息的方法
    def show(self):
        # __score可在类内访问, 但直接student1.__score会出错..
        print("Name: {}. Score: {}".format(self.name, self.__score))

# 实例化，创建对象
student1 = Student("jia.chen", 100)
student1.show()  # 打印 Name: John, Score: 100
# student1.__score  # 打印出错，该属性不能从外部访问. 


# 2. __call__
class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __call__(self, *args, **kwargs):
        print("执行实例方法call方法")

p = Person("test", 26)
print(p())   # 因为类中有__call__, 故可这样调用类 



# 3. __str__
class Person(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f"my name is {self.name} age is {self.age}"

p = Person("baozi", 26)
print(str(p))
print(p)    # 等价


# 4. __repr__
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



# 5. __getitem__()、__setitem__()、__delitem__() 类似字典的取值, 赋值, 删除 功能
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


# 6. @staticmethod: 装饰为静态方法, 就不需要传入self参数也可正常访问. 
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



# 7. @classmethod: 装饰为类方法, 无需实例化, 就可访问.
class Person(object):
    def __init__(self, name, age):
        self._name = name
        self._age = age

    @classmethod
    def static_talk(cls, name, age):  # 注意第一个参数: cls
        print(f"my name is {name} age is {age}")

print(Person.static_talk("jia.chen", 26))



# 8. 类的继承: 继承父类初始化, 子类中重写一些方法. 
# 创建父类学校成员SchoolMember
class SchoolMember:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def tell(self):
        # 打印个人信息
        print('Name:"{}" Age:"{}"'.format(self.name, self.age), end=" ")


# 创建子类老师 Teacher
class Teacher(SchoolMember):

    def __init__(self, name, age, salary):
        # 利用父类进行初始化
        SchoolMember.__init__(self, name, age) 
        # 新添加薪水 属性 
        self.salary = salary

    # 重写tell方法, 报告薪水.
    def tell(self):
        SchoolMember.tell(self)
        print('Salary: {}'.format(self.salary))


# 创建子类学生Student
class Student(SchoolMember):

    def __init__(self, name, age, score):
    # 利用父类进行初始化
        SchoolMember.__init__(self, name, age)
        self.score = score
    # 重写tell方法, 报告分数.
    def tell(self):
        SchoolMember.tell(self)  # 等价于, super().tell() 
        super().tell()  # super调用父类的方法. 等同于SchoolMember.tell(self)
        print('score: {}'.format(self.score))


teacher1 = Teacher("John", 44, "$60000")
student1 = Student("Mary", 12, 99)
teacher1.tell()  # 打印 Name:"John" Age:"44" Salary: $60000
student1.tell()  # Name:"Mary" Age:"12" score: 99


# 9. 



