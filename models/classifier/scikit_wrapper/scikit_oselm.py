import inspect

from models.tf_scikit_template import BaseTfScikitClassifier
from models.classifier.oselm import OSELM

class SciKitOSELM(BaseTfScikitClassifier):

    def __init__(self, 
                input_dim=None, 
                output_dim=None,
                hidden_num=None,
                batch_size=None, 
                flag_preprocess=False,
                tensorboard_path=None,

                X_valid=None,
                Y_valid=None,
                save_dir_path=None, 
                **kwargs
                ):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for key, val in values.items():
            setattr(self, key, val)

    def init_clf(self):
        self.clf = OSELM(self.input_dim,
                       self.output_dim,
                       hidden_num=self.hidden_num,
                       batch_size=self.batch_size,
                       flag_preprocess=self.flag_preprocess,
                       tensorboard_path=self.tensorboard_path)
 

    def fit(self, X, y=None, **kwargs):
        super().fit(**kwargs)

        self.clf.train(X, y,
                       X_valid=self.X_valid,
                       Y_valid=self.Y_valid,
                       save_dir_path=self.save_dir_path,
                       )

        return self

