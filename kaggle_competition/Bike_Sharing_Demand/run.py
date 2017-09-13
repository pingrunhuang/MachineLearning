'''
Practice based on http://brandonharris.io/kaggle-bike-sharing/
'''
import pandas as pd

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# feature engineering: the date value doesn't make sense, just keep the time
train_data["time"] = train_data["datetime"].str[11:20]
test_data["time"] = test_data["datetime"].str[11:20]
train_data["time"] = train_data["time"].str[:2].astype("int")
test_data["time"] = test_data["time"].str[:2].astype("int")

# transform the datatime into weekday
train_data["datetime"] = pd.to_datetime(train_data["datetime"])
train_data["weekday"] = train_data["datetime"].dt.dayofweek

test_data["datetime"] = pd.to_datetime(test_data["datetime"])
test_data["weekday"] = test_data["datetime"].dt.dayofweek

# validate some check to see which day of the week is more popular
print(train_data.groupby(['weekday'])['weekday', 'count'].mean())