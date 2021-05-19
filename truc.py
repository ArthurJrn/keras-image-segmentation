class MyClass():
    def __init__(self, b):
        self.a = "Hello"
        self.b = b

obj = MyClass("world")
obj.a = "truc"
obj.b = None

print(obj.a, obj.b)