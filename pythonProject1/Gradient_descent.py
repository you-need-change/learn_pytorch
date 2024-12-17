
class Foobar:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        fooward(self)


foobar = Foobar()
foobar(1, 2, 3)