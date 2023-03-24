from basemodel import BaseModel


class TestMolde(BaseModel):
    def __int__(self):
        super(TestMolde, self).__int__()

    def train_step(self, batch_size, opt):
        pass

    def test(self):
        pass

