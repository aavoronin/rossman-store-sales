from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import ExtraTreeRegressor
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from pandas import date_range
import os
import statsmodels.tools.eval_measures as sm
import xgboost as xgb


class models_provider():
    def __int__(self):
        pass

    def provide_XGBRegressor_s(self):
        """
                    'ElasticNet': ElasticNet(random_state=1),
                    'ElasticNet5000': ElasticNet(random_state=1, max_iter = 5000),
                    'Lasso': Lasso(random_state=1),
                    'Lasso5000': Lasso(random_state=1, max_iter = 5000),
                    'LinearRegression': LinearRegression(),
                    'RANSACRegressor': RANSACRegressor(random_state=1),
                    'RANSACRegressor500': RANSACRegressor(random_state=1, max_trials=500),
                    'RANSACRegressor2000': RANSACRegressor(random_state=1, max_trials=2000),
                    #'TheilSenRegressor': TheilSenRegressor(random_state=1),
                    #'TheilSenRegressor3000': TheilSenRegressor(random_state=1, max_iter=3000),
                    'SGDRegressor': SGDRegressor(random_state=1),
                    'SGDRegressor10000': SGDRegressor(random_state=1, max_iter=10000),
                    'MLPRegressor': MLPRegressor(random_state=1),
                    'MLPRegressor500': MLPRegressor(random_state=1, max_iter=500),
                    'MLPRegressor1000': MLPRegressor(random_state=1, max_iter=1000),
                    'HuberRegressor': HuberRegressor(),
                    'HuberRegressor1000': HuberRegressor(max_iter=1000),
                    'RandomForestRegressor': RandomForestRegressor(random_state=1),
                    'RandomForestRegressor500': RandomForestRegressor(random_state=1, n_estimators=500),
                    'RandomForestRegressor1000': RandomForestRegressor(random_state=1, n_estimators=1000),
                    'Ridge': Ridge(),
                    'ExtraTreesRegressor': ExtraTreesRegressor(),
                    'ARDRegression': ARDRegression(),
                    'AdaBoostRegressor': AdaBoostRegressor(random_state=1),
                    'BaggingRegressor': BaggingRegressor(),
                    'BayesianRidge': BayesianRidge(),
                    'CCA': CCA(),
                    'DecisionTreeRegressor': DecisionTreeRegressor(random_state=1),
                    'DummyRegressor': DummyRegressor(),
                    'ElasticNet': ElasticNet(random_state=1),
                    'ElasticNetCV': ElasticNetCV(random_state=1),
                    'ExtraTreeRegressor': ExtraTreeRegressor(random_state=1),
                    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=1),
                    #'GammaRegressor': GammaRegressor(),
                    #'GaussianProcessRegressor': GaussianProcessRegressor(random_state=1),
                    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=1),
                    'HistGradientBoostingRegressor': HistGradientBoostingRegressor(random_state=1),
                    'HuberRegressor': HuberRegressor(),
                    'IsotonicRegression': IsotonicRegression(),
                    'KNeighborsRegressor': KNeighborsRegressor(),

                    #'KernelRidge': KernelRidge(),
                    'Lars': Lars(random_state=1),
                    'LarsCV': LarsCV(),
                    'Lasso': Lasso(random_state=1),
                    'LassoCV': LassoCV(random_state=1),
                    'LassoLars': LassoLars(random_state=1),
                    'LassoLarsCV': LassoLarsCV(),
                    'LassoLarsIC': LassoLarsIC(),
                    'LinearRegression': LinearRegression(),
                    'LinearSVR': LinearSVR(random_state=1),
                    'MLPRegressor': MLPRegressor(random_state=1),
                    #'MultiOutputRegressor': MultiOutputRegressor(),
                    #'MultiTaskElasticNet': MultiTaskElasticNet(random_state=1),
                    #'MultiTaskElasticNetCV': MultiTaskElasticNetCV(random_state=1),
                    #'MultiTaskLasso': MultiTaskLasso(random_state=1),
                    #'MultiTaskLassoCV': MultiTaskLassoCV(random_state=1),
                    'NuSVR': NuSVR(),
                    'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
                    'OrthogonalMatchingPursuitCV': OrthogonalMatchingPursuitCV(),
                    'PLSCanonical': PLSCanonical(),
                    'PLSRegression': PLSRegression(),
                    'PassiveAggressiveRegressor': PassiveAggressiveRegressor(random_state=1),
                    'PoissonRegressor': PoissonRegressor(),
                    'QuantileRegressor': QuantileRegressor(),
                    'RANSACRegressor': RANSACRegressor(random_state=1),
                    'RadiusNeighborsRegressor': RadiusNeighborsRegressor(),
                    'RandomForestRegressor': RandomForestRegressor(random_state=1),
                    #'RegressorChain': RegressorChain(random_state=1),
                    'Ridge': Ridge(random_state=1),
                    'RidgeCV': RidgeCV(),
                    'SGDRegressor': SGDRegressor(random_state=1),
                    'SVR': SVR(),
                    #'StackingRegressor': StackingRegressor(),
                    # 'TheilSenRegressor': TheilSenRegressor(random_state=1),
                    # 'TransformedTargetRegressor': TransformedTargetRegressor(random_state=1),
                    # 'TweedieRegressor': TweedieRegressor(),
                    # 'VotingRegressor': VotingRegressor(),


                    'RandomForestRegressor': RandomForestRegressor(random_state=1),
                    'RandomForestRegressor500': RandomForestRegressor(random_state=1, n_estimators=500),
                    'RandomForestRegressor1000': RandomForestRegressor(random_state=1, n_estimators=1000),
                    'KNeighborsRegressor': KNeighborsRegressor(),
                    'BaggingRegressor(KNeighborsRegressor)': BaggingRegressor(
                        estimator=KNeighborsRegressor(), n_estimators=6,random_state=42),
                    'PoissonRegressor': PoissonRegressor(),
                    'MLPRegressor': MLPRegressor(random_state=1),
                    'MLPRegressor500': MLPRegressor(random_state=1, max_iter=500),
                    'MLPRegressor1000': MLPRegressor(random_state=1, max_iter=1000),
                    'ExtraTreeRegressor': ExtraTreeRegressor(random_state=1),
                    'LinearSVR': LinearSVR(random_state=1),
                    'NuSVR': NuSVR(),
                    'SGDRegressor': SGDRegressor(random_state=1),
                    'SVR': SVR(),
                    'BaggingRegressor(RandomForestRegressor100)': BaggingRegressor(
                        estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_estimators=6, random_state=42),
                    'BaggingRegressor(RandomForestRegressor1000)': BaggingRegressor(
                        estimator=RandomForestRegressor(n_estimators=1000, random_state=42), n_estimators=6, random_state=42),


                    #'KNeighborsRegressor': KNeighborsRegressor(),
                    'BaggingRegressor(KNeighborsRegressor)': BaggingRegressor(
                        estimator=KNeighborsRegressor(), n_estimators=6, random_state=42),
                    'LinearSVR': LinearSVR(random_state=1),
                    'NuSVR': NuSVR(),
                    'SGDRegressor': SGDRegressor(random_state=1),
                    'SVR': SVR(),
                    'BaggingRegressor(RandomForestRegressor100)': BaggingRegressor(
                        estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_estimators=6, random_state=42),
                    'BaggingRegressor(RandomForestRegressor1000)': BaggingRegressor(
                        estimator=RandomForestRegressor(n_estimators=1000, random_state=42), n_estimators=6, random_state=42),

                    #'RandomForestRegressor': [RandomForestRegressor(random_state=1)],
                    #'RandomForestRegressor500': [RandomForestRegressor(random_state=1, n_estimators=500)],
                    #'RandomForestRegressor1000': [RandomForestRegressor(random_state=1, n_estimators=1000)],

                    #'LinearRegression -> RandomForestRegressor': [LinearRegression(), RandomForestRegressor(random_state=1)],
                    #'LinearRegression -> RandomForestReressor500': [LinearRegression(), RandomForestRegressor(random_state=1, n_estimators=500)],
                    #'LinearRegression -> RandomForestRegressor1000': [LinearRegression(), RandomForestRegressor(random_state=1, n_estimators=1000)],

                    #'PoissonRegressor': [PoissonRegressor()],
                    #'LinearRegression -> PoissonRegressor': [LinearRegression(), PoissonRegressor()],

                    'RANSACRegressor500': [RANSACRegressor(random_state=1, max_trials=500)],
                    'RANSACRegressor1000': [RANSACRegressor(random_state=1, max_trials=1000)],
                    'RANSACRegressor2000': [RANSACRegressor(random_state=1, max_trials=2000)],

                    'LinearRegression -> RANSACRegressor500': [LinearRegression(), RANSACRegressor(random_state=1, max_trials=500)],
                    'LinearRegression -> RANSACRegressor1000': [LinearRegression(), RANSACRegressor(random_state=1, max_trials=1000)],
                    'LinearRegression -> RANSACRegressor2000': [LinearRegression(), RANSACRegressor(random_state=1, max_trials=2000)],

                    'MLPRegressor': [MLPRegressor(random_state=1)],
                    'MLPRegressor500': [MLPRegressor(random_state=1, max_iter=500)],
                    'MLPRegressor1000': [MLPRegressor(random_state=1, max_iter=1000)],
                    'ExtraTreeRegressor': [ExtraTreeRegressor(random_state=1)],

                    'LinearRegression -> MLPRegressor': [LinearRegression(), MLPRegressor(random_state=1)],
                    'LinearRegression -> MLPRegressor500': [LinearRegression(), MLPRegressor(random_state=1, max_iter=500)],
                    'LinearRegression -> MLPRegressor1000': [LinearRegression(), MLPRegressor(random_state=1, max_iter=1000)],
                    'LinearRegression -> ExtraTreeRegressor': [LinearRegression(), ExtraTreeRegressor(random_state=1)],


                    """

        list_of_models = {
            'LinearRegression': [LinearRegression()],
            'XGBRegressor': [xgb.XGBRegressor()],
            'LinearRegression -> XGBRegressor': [LinearRegression(), xgb.XGBRegressor()],
            'XGBRegressor(n_estimators=200, early_stopping_rounds=20)': [
                xgb.XGBRegressor(n_estimators=200, early_stopping_rounds=20)],
            'LinearRegression -> XGBRegressor(n_estimators=200, early_stopping_rounds=20)': [LinearRegression(),
                                                                                             xgb.XGBRegressor(
                                                                                                 n_estimators=200,
                                                                                                 early_stopping_rounds=20)],
            'XGBRegressor(n_estimators=100)': [xgb.XGBRegressor(n_estimators=100)],
            'LinearRegression -> XGBRegressor(n_estimators=100)': [LinearRegression(),
                                                                   xgb.XGBRegressor(n_estimators=100)],
            'XGBRegressor(max_depth=3)': [xgb.XGBRegressor(max_depth=3)],
            'LinearRegression -> XGBRegressor(max_depth=3)': [LinearRegression(), xgb.XGBRegressor(max_depth=3)],
            'XGBRegressor(learning_rate=0.1)': [xgb.XGBRegressor(learning_rate=0.1)],
            'LinearRegression -> XGBRegressor(learning_rate=0.1)': [LinearRegression(),
                                                                    xgb.XGBRegressor(learning_rate=0.1)],
            'XGBRegressor(subsample=0.8)': [xgb.XGBRegressor(subsample=0.8)],
            'LinearRegression -> XGBRegressor(subsample=0.8)': [LinearRegression(), xgb.XGBRegressor(subsample=0.8)],
            'XGBRegressor(colsample_bytree=0.5)': [xgb.XGBRegressor(colsample_bytree=0.5)],
            'LinearRegression -> XGBRegressor(colsample_bytree=0.5)': [LinearRegression(),
                                                                       xgb.XGBRegressor(colsample_bytree=0.5)],
            'XGBRegressor(gamma=0.2)': [xgb.XGBRegressor(gamma=0.2)],
            'LinearRegression -> XGBRegressor(gamma=0.2)': [LinearRegression(), xgb.XGBRegressor(gamma=0.2)],
            'XGBRegressor(alpha=0.5)': [xgb.XGBRegressor(alpha=0.5)],
            'LinearRegression -> XGBRegressor(alpha=0.5)': [LinearRegression(), xgb.XGBRegressor(alpha=0.5)],
            'XGBRegressor(objective="reg:squarederror")': [xgb.XGBRegressor(objective='reg:squarederror')],
            'LinearRegression -> XGBRegressor(objective="reg:squarederror")': [LinearRegression(), xgb.XGBRegressor(
                objective='reg:squarederror')],
            'XGBRegressor(eval_metric="rmse")': [xgb.XGBRegressor(eval_metric='rmse')],
            'LinearRegression -> XGBRegressor(eval_metric="rmse")': [LinearRegression(),
                                                                     xgb.XGBRegressor(eval_metric='rmse')],
            'XGBRegressor(early_stopping_rounds=10)': [xgb.XGBRegressor(early_stopping_rounds=10)],
            'LinearRegression -> XGBRegressor(early_stopping_rounds=10)': [LinearRegression(),
                                                                           xgb.XGBRegressor(early_stopping_rounds=10)],
            'XGBRegressor(n_estimators=50, max_depth=5)': [xgb.XGBRegressor(n_estimators=50, max_depth=5)],
            'LinearRegression -> XGBRegressor(n_estimators=50, max_depth=5)':
                [LinearRegression(), xgb.XGBRegressor(n_estimators=50, max_depth=5)],
            'XGBRegressor(learning_rate=0.05, subsample=0.9)': [xgb.XGBRegressor(learning_rate=0.05, subsample=0.9)],
            'LinearRegression -> XGBRegressor(learning_rate=0.05, subsample=0.9)': [LinearRegression(),
                                                                                    xgb.XGBRegressor(learning_rate=0.05,
                                                                                                     subsample=0.9)],
            'XGBRegressor(colsample_bytree=0.8, gamma=0.3)': [xgb.XGBRegressor(colsample_bytree=0.8, gamma=0.3)],
            'LinearRegression -> XGBRegressor(colsample_bytree=0.8, gamma=0.3)': [LinearRegression(),
                                                                                  xgb.XGBRegressor(colsample_bytree=0.8,
                                                                                                   gamma=0.3)],
            'XGBRegressor(objective="reg:linear", eval_metric="mae")': [
                xgb.XGBRegressor(objective='reg:linear', eval_metric='mae')],
            'LinearRegression -> XGBRegressor(objective="reg:linear", eval_metric="mae")': [LinearRegression(),
                                                                                            xgb.XGBRegressor(
                                                                                                objective='reg:linear',
                                                                                                eval_metric='mae')],
            'XGBRegressor(max_depth=4, learning_rate=0.2, subsample=0.7)': [
                xgb.XGBRegressor(max_depth=4, learning_rate=0.2, subsample=0.7)],
            'LinearRegression -> XGBRegressor(max_depth=4, learning_rate=0.2, subsample=0.7)': [LinearRegression(),
                                                                                                xgb.XGBRegressor(
                                                                                                    max_depth=4,
                                                                                                    learning_rate=0.2,
                                                                                                    subsample=0.7)],
            'XGBRegressor(colsample_bytree=0.6, gamma=0.1, objective="reg:squarederror", eval_metric="rmse")': [
                xgb.XGBRegressor(colsample_bytree=0.6, gamma=0.1, objective='reg:squarederror', eval_metric='rmse')],
            'LinearRegression -> XGBRegressor(colsample_bytree=0.6, gamma=0.1, objective="reg:squarederror", eval_metric="rmse")': [
                LinearRegression(),
                xgb.XGBRegressor(colsample_bytree=0.6, gamma=0.1, objective='reg:squarederror', eval_metric='rmse')],
            'XGBRegressor(reg_lambda=0.1)': [xgb.XGBRegressor(reg_lambda=0.1)],
            'LinearRegression -> XGBRegressor(reg_lambda=0.1)': [LinearRegression(), xgb.XGBRegressor(reg_lambda=0.1)],
            'XGBRegressor(alpha=0.1, reg_lambda=0.05)': [xgb.XGBRegressor(alpha=0.1, reg_lambda=0.05)],
            'LinearRegression -> XGBRegressor(alpha=0.1, reg_lambda=0.05)': [LinearRegression(),
                                                                             xgb.XGBRegressor(alpha=0.1,
                                                                                              reg_lambda=0.05)],

            'RandomForestRegressor': [RandomForestRegressor(random_state=1)],
            'RandomForestRegressor500': [RandomForestRegressor(random_state=1, n_estimators=500)],
            'RandomForestRegressor1000': [RandomForestRegressor(random_state=1, n_estimators=1000)],

            'LinearRegression -> RandomForestRegressor': [LinearRegression(), RandomForestRegressor(random_state=1)],
            'LinearRegression -> RandomForestReressor500': [LinearRegression(), RandomForestRegressor(random_state=1, n_estimators=500)],
            'LinearRegression -> RandomForestRegressor1000': [LinearRegression(), RandomForestRegressor(random_state=1, n_estimators=1000)],

        }



        regressor_classes = dict(
            [(f'KNeighborsRegressor {n_neighbors} {weights} {algorithm}',
              [KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)])
             for n_neighbors in range(5, 20, 5)
             for weights in ['uniform', 'distance']
             for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']]
        )

        # for model in regressor_classes.keys():
        #    list_of_models[model] = regressor_classes[model]

        regressor_classes = dict(
            [(f'LinearRegression -> KNeighborsRegressor {n_neighbors} {weights} {algorithm}',
              [LinearRegression(), KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)])
             for n_neighbors in range(5, 20, 5)
             for weights in ['uniform', 'distance']
             for algorithm in ['auto', 'ball_tree', 'kd_tree', 'brute']]
        )

        # for model in regressor_classes.keys():
        #    list_of_models[model] = regressor_classes[model]

        return list_of_models