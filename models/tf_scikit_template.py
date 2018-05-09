from sklearn.base import BaseEstimator, ClassifierMixin


def check_clf(instance):
    flag = False
    if 'clf' in instance.__dict__:
        if instance.clf is not None:
            flag = True
    return flag


class BaseTfScikitClassifier(BaseEstimator, ClassifierMixin):

    def init_clf(self):
        pass

    def fit(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

        if not check_clf(self):
            self.init_clf()

    def predict(self, X, y=None, batch_size=64):
        return self.clf.predict(X, batch_size)

    def evaluate(self, X, y=None, batch_size=64):
        """
        return Y_pred, accuracy, loss
        """
        return self.clf.evaluate(X, y, batch_size)

    def score(self, X, y=None, batch_size=64):
        _, accuracy, _ = self.clf.evaluate(X, y, batch_size)
        return accuracy

    def load(self, path):
        if not check_clf(self):
            print("init_clf()")
            self.init_clf()
        self.clf.load(path)

    def save(self, path):
        self.clf.save(path)

