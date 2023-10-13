from scipy import stats
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.tree import ExtraTreeRegressor
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from pandas import date_range
import os
import statsmodels.tools.eval_measures as sm
import xgboost as xgb

# Model 2

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor, PoissonRegressor, SGDRegressor

from BoostedHybrid import BoostedHybrid
from models_provider import models_provider
from predict_base import predict_base
from utils import sanitize_filename


def days_since_promo(Promo2, Promo2SinceWeek, Promo2SinceYear, month_string, date):
    promo_never = 100000
    if Promo2 == 0:
        return promo_never

    month_list = []

    month_mapping = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }

    months = month_string.split(',')

    for month in months:
        month_number = month_mapping.get(month.strip())
        if month_number:
            month_list.append(month_number)

    first_day = datetime.date(Promo2SinceYear, 1, 1)

    # Calculate the number of days to add based on the week number
    days_to_add = (Promo2SinceWeek - 1) * 7

    # Add the number of days to the first day of the year
    promo_since_date = first_day + datetime.timedelta(days=days_to_add)

    if pd.Timestamp(date) < pd.Timestamp(promo_since_date):
        return promo_never

    year = date.year
    month = date.month

    for _ in range(12):
        first_day = datetime.date(year, month, 1)
        if month in month_list:
            if first_day > promo_since_date:
                days = (pd.Timestamp(date) - pd.Timestamp(first_day)).days
                if days < 1:
                    days = 1
                return days
            else:
                return promo_never

        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1
    return promo_never




class predict_rossman_sales(predict_base):
    def __init__(self):
        super().__init__()
        self.project_name = 'predict rossman sales'
        self.input_dir = 'rossmann-store-sales/input'
        self.output_dir = 'rossmann-store-sales/output'
        self.sales_cache_file = os.path.join(self.input_dir, 'data.pkl')

    def load_data(self):
        super().load_data()

        for dirname, _, filenames in os.walk(self.input_dir):
            for filename in filenames:
                print(os.path.join(dirname, filename))

        self.sales0 = pd.read_csv(os.path.join(self.input_dir, 'train.csv'), parse_dates=["Date"],
            dtype = {
                'store': 'int32',
                'DayOfWeek': 'int8',
                'Sales': 'float32',
                'Customers': 'int32',
                'Open': 'int32',
                'Promo': 'int32',
                'StateHoliday': 'category',
                'SchoolHoliday': 'int8',
            })

        self.ID_INCREMENT = 1000000000
        self.test = pd.read_csv(os.path.join(self.input_dir, 'test.csv'), parse_dates=["Date"])
        self.test['Open'].fillna(0, inplace=True)
        self.test['Id'] = self.test['Id'].astype('int32')
        self.test['Store'] = self.test['Store'].astype('int32')
        self.test['DayOfWeek'] = self.test['DayOfWeek'].astype('int8')
        self.test['Open'] = self.test['Open'].astype('int32')
        self.test['Promo'] = self.test['Promo'].astype('int32')
        self.test['StateHoliday'] = self.test['StateHoliday'].astype('category')
        self.test['SchoolHoliday'] = self.test['SchoolHoliday'].astype('int8')
        self.test['Sales'] = 0
        self.test['Sales'] = self.test['Sales'].astype('float32')
        self.test.index = self.test.index.map(lambda x: x + self.ID_INCREMENT)

        self.sales = pd.concat([
            self.sales0[['Store', 'DayOfWeek', 'Date', 'Sales', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']],
            self.test[['Store', 'DayOfWeek', 'Date', 'Sales', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']]])
        #sales = sales.reindex(np.arange(self.sales0.shape[0] + self.test.shape[0]))

        self.stores = pd.read_csv(os.path.join(self.input_dir, 'store.csv'))

    def design_features(self):
        super().design_features()
        cache_loaded = False
        try:
            if os.path.exists(self.sales_cache_file):
                sales_from_cache = pd.read_pickle(self.sales_cache_file)
                if sales_from_cache.shape[0] == self.sales.shape[0]:
                    cache_loaded = True
                    del self.sales
                    self.sales = sales_from_cache
        except Exception as e:
            cache_loaded = False

        if not cache_loaded:
            self.sales['DayOfWeek'] = self.sales['Date'].dt.dayofweek
            self.sales['WeekISO'] = self.sales['Date'].apply(lambda x: x.isocalendar()[1])
            self.sales['YearISO'] = self.sales['Date'].apply(lambda x: x.isocalendar()[0])
            self.sales['Month'] = self.sales['Date'].dt.month
            self.sales['Year'] = self.sales['Date'].dt.year

            self.stores['CompetitionOpenSinceMonth'] = pd.to_numeric(self.stores['CompetitionOpenSinceMonth'], errors='coerce',
                                                                downcast='integer')
            self.stores['CompetitionOpenSinceMonth'] = self.stores['CompetitionOpenSinceMonth'].astype('Int32')
            self.stores['CompetitionOpenSinceYear'] = pd.to_numeric(self.stores['CompetitionOpenSinceYear'], errors='coerce',
                                                               downcast='integer')
            self.stores['CompetitionOpenSinceYear'] = self.stores['CompetitionOpenSinceYear'].astype('Int32')
            self.stores['CompetitionOpenSinceDate'] = self.stores.apply(
                lambda x: pd.to_datetime(str(x['CompetitionOpenSinceYear']) + '-' +
                                         str(x['CompetitionOpenSinceMonth']) + '-01',
                                         errors='coerce'), axis=1)
            self.stores['CompetitionOpenSinceDate'] = self.stores['CompetitionOpenSinceDate'].astype('datetime64[ns]')

            self.stores['Promo2SinceWeek'] = self.stores['Promo2SinceWeek'].astype('Int32')
            self.stores['Promo2SinceYear'] = self.stores['Promo2SinceYear'].astype('Int32')

            #columns_to_add = ['StoreType', 'Assortment', 'CompetitionDistance', 'CompetitionOpenSinceDate', 'Promo2',
            #                  'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'Days Since Promo']

            # 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'
            # Perform the left join to add the fields to the "sales" dataframe
            self.sales = pd.merge(self.sales, self.stores, on='Store', how='left')

            self.sales = self.one_hot_encode_df_inplace(self.sales, 'StoreType')
            self.sales = self.one_hot_encode_df_inplace(self.sales, 'Assortment')
            self.sales = self.one_hot_encode_df_inplace(self.sales, 'DayOfWeek')
            self.sales = self.one_hot_encode_df_inplace(self.sales, 'StateHoliday')
            self.sales = self.one_hot_encode_df_inplace(self.sales, 'SchoolHoliday')

            self.sales = self.df_boxcox_column(self.sales, 'CompetitionDistance')

            self.sales['Competition'] = np.where(
                self.sales['CompetitionOpenSinceDate'].isnull() |
                (self.sales['CompetitionOpenSinceDate'] > self.sales['Date']),
                False,
                True
            )
            self.sales.loc[~self.sales['Competition'], 'CompetitionDistance'] = 1000000
            self.sales['Days Since Promo'] = self.sales.apply(lambda x:
                days_since_promo(x['Promo2'], x['Promo2SinceWeek'], x['Promo2SinceYear'], x['PromoInterval'], x['Date']), axis=1)

            # Perform logarithm transformation on 'Competition' column
            self.sales['CompetitionDistanceLogBuckets'] = np.log(self.sales['CompetitionDistance'])
            #self.sales['DaysSincePromoBuckets'] = np.log(self.sales['Days Since Promo'])

            # Apply Box-Cox transformation to spread values into 6 buckets
            self.sales['CompetitionDistanceLogBuckets'], _ = stats.boxcox(self.sales['CompetitionDistanceLogBuckets'])
            self.sales['DaysSincePromoBuckets'], _ = stats.boxcox(self.sales['Days Since Promo'])

            # Replace values with the number of the bucket
            self.sales['CompetitionDistanceLogBuckets'] = pd.qcut(self.sales['CompetitionDistanceLogBuckets'], q=10, labels=False, duplicates='drop')
            self.sales['DaysSincePromoBuckets'] = pd.qcut(self.sales['DaysSincePromoBuckets'], q=10, labels=False, duplicates='drop')

            # Assuming 'df' is your DataFrame
            min_date = self.sales['Date'].min()
            max_date = self.sales['Date'].max()
            dr_index = date_range(min_date, freq="D", periods=(max_date - min_date).days + 1)

            fourierW = CalendarFourier(freq="W", order=6)
            fourierA = CalendarFourier(freq="A", order=12)
            dp = DeterministicProcess(
                index=dr_index,
                constant=True,
                order=2,
                seasonal=True,
                additional_terms=[fourierW, fourierA],
                drop=True,
            )

            sales_trends = dp.in_sample()

            sales_trends['Date'] = dr_index.to_frame()
            self.sales = pd.merge(self.sales, sales_trends, left_on='Date', right_index=True, how='left')
            self.sales.to_pickle(self.sales_cache_file)

        self.sales.drop(columns=['Date_x', 'Date_y', 'PromoInterval', 'CompetitionOpenSinceDate',
                             'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2SinceWeek',
                             'Promo2SinceYear'], inplace=True)

        self.y = self.sales0[['Sales']]
        self.X = self.sales
        self.X.drop(columns=['Sales', 'Store', 'WeekISO', 'YearISO', 'Month', 'Year'], inplace=True)
        self.X.drop(columns=['CompetitionDistanceLogBuckets', 'Competition', 'const'], inplace=True)

        print(self.X.dtypes)
        print(self.X.columns)

        """
        X_ = self.X.iloc[:self.y.shape[0]].copy()

        for column in X_.columns:
            X_[column].fillna("Unknown", inplace=True)

        for colname in X_.select_dtypes("object"):
            X_[colname].fillna("Unknown", inplace=True)
            X_[colname], _ = X_[colname].factorize()

        discrete_features = (X_.dtypes == int)
        mi_scores = self.make_mi_scores(X_, self.y, discrete_features)
        mi_scores2 = mi_scores.where(mi_scores > 0)
        mi_scores2 = mi_scores2.dropna()

        for index, value in mi_scores2.items():
            print(f"{index:24}: {value}")
        """


    def init_list_of_models(self):
        super().init_list_of_models()
        self.list_of_models = models_provider().provide_XGBRegressor_s()

    def done(self):
        super().done()


    def init_Xy(self, ri):
        yy = pd.DataFrame(self.y['Sales'], columns=['Sales'])
        if len(ri.models) == 1:
            XX = self.X.iloc[:self.sales0.shape[0]]
            return ([XX], yy)
        if len(ri.models) == 2:
            XX = self.X.iloc[:self.sales0.shape[0]]
            return ([XX, XX], yy)

    def init_train_test_Xy(self):
        self.percent = 80

        min_date = self.sales0['Date'].min()
        max_date = self.sales0['Date'].max()

        days = (max_date - min_date + 1).days

        self.train_indexes
        XX = self.X.iloc[:self.sales0.shape[0]]
        yy = self.y['Sales']
        return (XX, yy)

    def predict(self, model, model_name):
        if isinstance(model, BoostedHybrid):
            self.y_pred = pd.DataFrame(model.predict(self.X, self.X))
            if 'Sales' in self.y_pred.columns:
                self.y_pred.rename(columns={'Sales': 'Sales Pred'}, inplace=True)
        else:
            self.y_pred = pd.DataFrame(model.predict(self.X), columns=['Sales Pred'])
        #self.y_pred = pd.DataFrame(model.predict(self.X.iloc[:self.sales0.shape[0]]), columns=['Sales Pred'])
        #self.y_pred[['Store', 'Date', ]] = self.sales[['Store', 'Date']]
        #self.y_pred[['DayOfWeek_6', 'Open']] = self.X[['DayOfWeek_6', 'Open']]
        #self.y_pred['Store'] = self.sales['Store']
        # y_pred.loc[y_pred['DayOfWeek_6'], 'Sales Pred'] = 0.0
        # y_pred.loc[y_pred['Open'] == 0, 'Sales Pred'] = 0.0
        existing = self.y['Sales']
        #predicted = self.y_pred.iloc[:self.sales0.shape[0]]['Sales Pred']

        self.y_pred.loc[self.X['DayOfWeek_6'], 'Sales Pred'] = 0.0
        self.y_pred.loc[self.X['Open'] == 0, 'Sales Pred'] = 0.0

        predicted = self.y_pred.iloc[:self.sales0.shape[0]]
        rmspe = sm.rmspe(existing, predicted['Sales Pred'])
        print(f"{model_name} RMSPE:, {rmspe}")

        """
        test_X = self.X.iloc[self.sales0.shape[0]:].copy()
        #test_X.index -= self.ID_INCREMENT
        test_X.reset_index(drop=True, inplace=True)
        y_test_pred = pd.DataFrame(model.predict(test_X), columns=['Sales Pred'])
        y_test_pred.reset_index(drop=True, inplace=True)

        y_test_pred.loc[test_X['DayOfWeek_6'], 'Sales Pred'] = 0.0
        y_test_pred.loc[test_X['Open'] == 0, 'Sales Pred'] = 0.0

        
        """
        y_test_pred = self.y_pred.iloc[self.sales0.shape[0]:].copy()
        y_test_pred.reset_index(drop=True, inplace=True)

        X_test = self.test.copy()
        X_test.reset_index(drop=True, inplace=True)

        self.submission = pd.concat([X_test['Id'].astype('int32'), y_test_pred['Sales Pred']], axis=1)
        self.submission.rename(columns={'Sales Pred': 'Sales'}, inplace=True)

        file_name = sanitize_filename(f'submission_{model_name}_{rmspe}.csv')
        self.submission.to_csv(os.path.join(self.output_dir, file_name), index=False)

        del self.y_pred
        del self.submission
        del X_test
        del y_test_pred


