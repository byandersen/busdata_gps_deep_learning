import numpy as np
import pandas as pd
import torch


class LocationBusDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(
            csv_file,
            index_col=False,
            parse_dates=['last_modified1', 'last_modified2'],
        )

        # normalized latitude and longitude
        min_latitude, max_latitude = 54.7, 54.9
        min_longitude, max_longitude = 9.1, 9.6
        self.data['n_latitude1'] = (self.data['latitude1'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_latitude2'] = (self.data['latitude2'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_longitude1'] = (self.data['longitude1'] - min_longitude) / (max_longitude - min_longitude)
        self.data['n_longitude2'] = (self.data['longitude2'] - min_longitude) / (max_longitude - min_longitude)

        self.data['group_cat'] = self.data['group'].astype('category').cat.codes
        self.num_groups = self.data['group_cat'].nunique()

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # normalized pos1 and pos2
        data = row[['n_latitude1', 'n_latitude2', 'n_longitude1', 'n_longitude2']].to_numpy()

        # add route group as one hot
        group = np.zeros(self.num_groups, dtype=np.float64)
        group[row['group_cat']] = 1.0

        data = np.append(data, group)
        data = np.float64(data)

        # label is the time diff in minutes
        labels = [(row['last_modified2'] - row['last_modified1']).total_seconds() / 60]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)


class LocationTimeBusDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(
            csv_file,
            index_col=False,
            parse_dates=['last_modified1', 'last_modified2'],
        )

        # normalized latitude and longitude
        min_latitude, max_latitude = 54.7, 54.9
        min_longitude, max_longitude = 9.1, 9.6
        self.data['n_latitude1'] = (self.data['latitude1'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_latitude2'] = (self.data['latitude2'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_longitude1'] = (self.data['longitude1'] - min_longitude) / (max_longitude - min_longitude)
        self.data['n_longitude2'] = (self.data['longitude2'] - min_longitude) / (max_longitude - min_longitude)

        self.data['n_hour'] = self.data['last_modified1'].dt.hour / 24
        self.data['n_minute'] = self.data['last_modified1'].dt.minute / 60
        self.data['n_second'] = self.data['last_modified1'].dt.second / 60
        self.data['day_of_week'] = self.data['last_modified1'].dt.dayofweek

        self.data['group_cat'] = self.data['group'].astype('category').cat.codes
        self.num_groups = self.data['group_cat'].nunique()

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # normalized pos1 and pos2
        data = row[
            ['n_latitude1', 'n_latitude2', 'n_longitude1', 'n_longitude2', 'n_hour', 'n_minute', 'n_second']].to_numpy()

        day_of_week = np.zeros(7, dtype=np.float64)
        day_of_week[row['day_of_week']] = 1.0
        data = np.append(data, day_of_week)

        # add route group as one hot
        group = np.zeros(self.num_groups, dtype=np.float64)
        group[row['group_cat']] = 1.0

        data = np.append(data, group)
        data = np.float64(data)

        # label is the time diff in minutes
        labels = [(row['last_modified2'] - row['last_modified1']).total_seconds() / 60]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)


class LocationTimeAvgSpeedBusDataLoader(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(
            csv_file,
            index_col=False,
            parse_dates=['last_modified1', 'last_modified2'],
        )

        # normalized latitude and longitude
        min_latitude, max_latitude = 54.7, 54.9
        min_longitude, max_longitude = 9.1, 9.6
        self.data['n_latitude1'] = (self.data['latitude1'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_latitude2'] = (self.data['latitude2'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_longitude1'] = (self.data['longitude1'] - min_longitude) / (max_longitude - min_longitude)
        self.data['n_longitude2'] = (self.data['longitude2'] - min_longitude) / (max_longitude - min_longitude)

        self.data['n_hour'] = self.data['last_modified1'].dt.hour / 24
        self.data['n_minute'] = self.data['last_modified1'].dt.minute / 60
        self.data['n_second'] = self.data['last_modified1'].dt.second / 60
        self.data['day_of_week'] = self.data['last_modified1'].dt.dayofweek

        self.data['n_avg_speed'] = (self.data['avg_speed'] * 6371 * 60).replace([np.inf, -np.inf, np.nan], 0)

        self.data['group_cat'] = self.data['group'].astype('category').cat.codes
        self.num_groups = self.data['group_cat'].nunique()

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # normalized pos1 and pos2
        data = row[['n_latitude1', 'n_latitude2', 'n_longitude1', 'n_longitude2', 'n_hour', 'n_minute', 'n_second',
                    'n_avg_speed']].to_numpy()

        day_of_week = np.zeros(7, dtype=np.float64)
        day_of_week[row['day_of_week']] = 1.0
        data = np.append(data, day_of_week)

        # add route group as one hot
        group = np.zeros(self.num_groups, dtype=np.float64)
        group[row['group_cat']] = 1.0

        data = np.append(data, group)
        data = np.float64(data)

        # label is the time diff in minutes
        labels = [(row['last_modified2'] - row['last_modified1']).total_seconds() / 60]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)



class LocationTimeWithoutGroupBusDataLoader(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data = df.copy()

        # normalized latitude and longitude
        min_latitude, max_latitude = 54.7, 54.9
        min_longitude, max_longitude = 9.1, 9.6
        self.data['n_latitude1'] = (self.data['latitude1'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_latitude2'] = (self.data['latitude2'] - min_latitude) / (max_latitude - min_latitude)
        self.data['n_longitude1'] = (self.data['longitude1'] - min_longitude) / (max_longitude - min_longitude)
        self.data['n_longitude2'] = (self.data['longitude2'] - min_longitude) / (max_longitude - min_longitude)

        self.data['n_hour'] = self.data['last_modified1'].dt.hour / 24
        self.data['n_minute'] = self.data['last_modified1'].dt.minute / 60
        self.data['n_second'] = self.data['last_modified1'].dt.second / 60
        self.data['day_of_week'] = self.data['last_modified1'].dt.dayofweek

        self.data['n_avg_speed'] = (self.data['avg_speed'] * 6371 * 60).replace([np.inf, -np.inf, np.nan], 0)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # normalized pos1 and pos2
        data = row[['n_latitude1', 'n_latitude2', 'n_longitude1', 'n_longitude2', 'n_hour', 'n_minute', 'n_second',
                    'n_avg_speed']].to_numpy()

        day_of_week = np.zeros(7, dtype=np.float64)
        day_of_week[row['day_of_week']] = 1.0
        data = np.append(data, day_of_week)
        data = np.float64(data)

        # label is the time diff in minutes
        labels = [(row['last_modified2'] - row['last_modified1']).total_seconds() / 60]

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
