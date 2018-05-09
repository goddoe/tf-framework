import inspect

from models.tf_scikit_template import BaseTfScikitClassifier
from models.classifier.lstm import LSTM

class SciKitLSTM(BaseTfScikitClassifier):

    def __init__(self, 
                input_dim=None, 
                output_dim=None,
                lstm_info_dict=None,
                fc_layer_info_dict_list=None,
                output_layer_info_dict=None,
                optimizer = None,
                cost_function=None,
                flag_preprocess=False,
                tensorboard_path=None,

                X_valid=None,
                Y_valid=None,
                batch_size=32, 
                n_epoch=300, 
                learning_rate=0.1, 
                reg_lambda=0.,
                patience=100,
                verbose_interval=20,
                save_dir_path=None, 
                **kwargs
                ):

        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for key, val in values.items():
            setattr(self, key, val)

    def init_clf(self):
        self.clf = LSTM(self.input_dim,
                       self.output_dim,
                       self.lstm_info_dict,
                       self.fc_layer_info_dict_list,
                       self.output_layer_info_dict,
                       self.optimizer,
                       self.cost_function,
                       self.flag_preprocess,
                       tensorboard_path=self.tensorboard_path)


    def fit(self, X, y=None, **kwargs):
        super().fit(**kwargs)

        self.clf.train(X, y,
                       X_valid=self.X_valid,
                       Y_valid=self.Y_valid,
                       batch_size=self.batch_size,
                       n_epoch=self.n_epoch,
                       learning_rate=self.learning_rate,
                       reg_lambda=self.reg_lambda,
                       patience=self.patience,
                       verbose_interval=self.verbose_interval,
                       save_dir_path=self.save_dir_path,
                       )

        return self


