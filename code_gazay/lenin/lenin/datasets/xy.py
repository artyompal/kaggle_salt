class XY:
    fields = ['x']
    target = 'y'

    def __init__(self, data):
        self.data = data
        self.train = list(range(len(data)))

    def x(self, record, augmentor=lambda x: x):
        return augmentor(self.data[record][0])

    def y(self, record):
        return self.data[record][1]
