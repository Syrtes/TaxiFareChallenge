# imports
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
import pandas as pd
import numpy as np
from memoized_property import memoized_property
import mlflow
from  mlflow.tracking import MlflowClient


MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "[AG]"
EXPERIMENT_NAME = f"[Taxifare] [BX] [#516] {myname}"




class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.X = X
        self.y = y
        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        '''returns a pipelined model'''

        pipe_distance = make_pipeline(DistanceTransformer(), RobustScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'), StandardScaler())

        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preprocessing = ColumnTransformer([('time', pipe_time, time_cols),
                                          ('distance', pipe_distance, dist_cols)]
                                          )

        pipe_final = Pipeline(steps=[('preprocessor', preprocessing),
                                ('regressor', LinearRegression())])

        self.mlflow_client.log_param(self.mlflow_run.info.run_id, "model", "linear")
        self.pipeline = pipe_final


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline_trained = self.pipeline.fit(self.X, self.y)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline_trained.predict(X_test)
        rmse = np.sqrt(((y_pred - y_test)**2).mean())

        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, "rmse", rmse)

        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df_clean = clean_data(df)
    # set X and y
    y = df.fare_amount
    X = df.drop(columns='fare_amount')
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    # train
    model = Trainer(X_train,y_train)
    model.run()
    print(f"RMSE is {model.evaluate(X_test, y_test)}")



