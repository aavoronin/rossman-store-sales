import datetime
import multiprocessing
import threading
import pandas as pd
from scipy import stats
from multiprocessing.pool import ThreadPool

from sklearn.feature_selection import mutual_info_regression

from BoostedHybrid import BoostedHybrid
from run_info import run_info


def run_model_fit(models, X, y):
    if len(models) == 1 and len(X) == 1:
        models[0].fit(X[0], y)
        return models[0]
    if len(models) == 2 and len(X) == 2:
        double_model = BoostedHybrid(models[0], models[1])
        double_model.fit(X[0], X[1], y)
        return double_model
    models[0].fit(X[0], y)
    return models[0]

class predict_base:
    def __init__(self):
        self.project_name = 'predict_base'
        self.list_of_models = []
        self.timeout = 600
        self.processes = 5

    def load_data(self):
        print(f'{datetime.datetime.now()} {self.project_name} load_data')

    def design_features(self):
        print(f'{datetime.datetime.now()} {self.project_name} design_features')

    def one_hot_encode_df_inplace(self, df, colname):
        # Perform one-hot encoding on the specified column
        if colname not in df.columns:
            return df
        encoded_cols = pd.get_dummies(df[colname], prefix=colname, drop_first=True)
        df.drop(colname, axis=1, inplace=True)
        # print(encoded_cols)
        df = pd.concat([df, encoded_cols], axis=1)
        return df

        # Remove any unused space from the dataframe
        # df.columns = df.columns.str.strip()
        # df.reset_index(drop=True, inplace=True)


    def df_boxcox_column(self, df, colname):
        if colname in df.columns:
            return df
        a = stats.boxcox(df[colname])
        df[colname + '_boxcox'] = a[0]
        return df

    def init_list_of_models(self):
        print(f'{datetime.datetime.now()} {self.project_name} init_list_of_models')

    def get_run_infos(self):
        return [run_info(regressor_name, self.list_of_models[regressor_name]) for regressor_name in self.list_of_models.keys()]

    def done(self):
        print(f'{datetime.datetime.now()} {self.project_name} done')

    def run_models(self):
        #self.pool = ThreadPool(processes=self.processes)
        self.pool = multiprocessing.Pool(processes=self.processes)

        task_collection = []

        for ri in self.get_run_infos():
            try:
                ri.start_time = datetime.datetime.now()
                print(f'{datetime.datetime.now()} {ri.get_run_name()} prediction start')
                ri.models = self.list_of_models[ri.model_class_name]
                ri.XXs, ri.yy = self.init_Xy(ri)
                async_result = self.pool.apply_async(func=run_model_fit, args=(ri.models, ri.XXs, ri.yy))  # tuple of args for foo
                task_collection.append((async_result, ri.models, ri))
            except TimeoutError as e:
                print(f'{datetime.datetime.now()} {ri.get_run_name()} timed out')
            except Exception as e:
                print(e)

        for async_result, model, ri in task_collection:
            try:
                final_model = async_result.get(timeout=6000)
                self.predict(final_model, ri.get_run_name())
                time_diff_str = self.timediff(datetime.datetime.now() - ri.start_time)
                print(f'{datetime.datetime.now()} {ri.get_run_name()} prediction complete ({time_diff_str})')
                del ri.XXs
                del ri.yy
                del model
            except TimeoutError as e:
                print(f'{datetime.datetime.now()} {ri.get_run_name()} timed out')
            except Exception as e:
                print(e)

        del self.pool

    def init_Xy(self):
        raise NotImplementedError("This method is should be overwritten")

    def predict(self, model, model_name):
        raise NotImplementedError("This method is should be overwritten")

    def timediff(self, time_difference):
        hours = time_difference.seconds // 3600
        minutes = (time_difference.seconds // 60) % 60
        seconds = time_difference.seconds % 60
        return f'{hours:02}:{minutes:02}:{seconds:02}'

    def make_mi_scores(self, X, y, features):
        mi_scores = mutual_info_regression(X, y, discrete_features=features)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        return mi_scores