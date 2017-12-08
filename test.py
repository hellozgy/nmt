class A:

    def __init__(self):
        self.a = 10
        self.B = B(self.show)

    def f(self):
        print(self.B.show_b())

    def show(self, args):
        return self.a+args

class B:
    def __init__(self, show):
        self.show = show
        self.t = 2

    def show_b(self):
        return self.show(self.t)

a=A()
a.f()