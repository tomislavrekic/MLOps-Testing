"""Helper functions for RF on titanic dataset"""
import numpy as np
from pandas import DataFrame
from sklearn.metrics import accuracy_score


titles_map = {
    'Mr' :         'Mr',
    'Mme':         'Mrs',
    'Ms':          'Mrs',
    'Mrs' :        'Mrs',
    'Master' :     'Master',
    'Mlle':        'Miss',
    'Miss' :       'Miss',
    'Capt':        'Officer',
    'Col':         'Officer',
    'Major':       'Officer',
    'Dr':          'Officer',
    'Rev':         'Officer',
    'Jonkheer':    'Royalty',
    'Don':         'Royalty',
    'Sir' :        'Royalty',
    'Countess':    'Royalty',
    'Dona':        'Royalty',
    'Lady' :       'Royalty'
}

class DataPreprocessor():
    """Data preprocessing for titanic dataset
    """
    def __init__(self) -> None:
        #self.col_index_set = False
        pass

    def extract_title(self, names):
        '''Extracts the title from the passenger names.'''

        return names.str.extract(' ([A-Za-z]+)\.', expand=False).map(titles_map)

    def preprocess_dataset(self, data : DataFrame, test=False):
        """Data preprocessing for titanic dataset

        Args:
            data (DataFrame): Unprocessed dataset
            test (bool, optional): Is dataset a test dataset. Defaults to False.

        Returns:
            DataFrame, DataFrame: X and Y split, if test=False
            DataFrame: X , if test=True
        """

        in_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'Embarked', 'Fare']
        out_features = ['Survived']

        #train_df['Title'] = train_df['Name'].map(lambda x: x.split(',')[1].split('.')[0])

        in_df = data.dropna(subset=["Embarked"])
        if not test:
            out_y = in_df[out_features]
        in_df = in_df[in_features]
        in_df.loc[in_df["Age"].isnull(), "Age"] = in_df["Age"].mean()
        in_df.loc[in_df["Fare"].isnull(), "Fare"] = in_df["Fare"].mean()
        in_df['Male'] = in_df['Sex'].map(lambda x: x=="male")
        in_df['Title'] = self.extract_title(data['Name'])

        titles = set(titles_map.values())
        in_df = in_df.head(5)
        for title in titles:
            in_df['is_' + title] = in_df['Title'].map(lambda x: x==title)

        for embarked in ['C', 'Q', 'S']:
            in_df['Embarked_' + embarked] = in_df['Embarked'].map(
                lambda x: x==embarked)

        for p_class in [1, 2, 3]:
            in_df['Pclass_' + str(p_class)] = in_df['Pclass'].map(
                lambda x: x==p_class)

        in_df['FamilySize'] = in_df['SibSp'] + in_df['Parch']
        in_df['FarePerPerson'] = in_df['Fare'] / (1 + in_df['FamilySize'])

        in_df.drop(columns="Title", inplace=True)
        in_df.drop(columns="Embarked", inplace=True)
        in_df.drop(columns="Sex", inplace=True)
        in_df.drop(columns="Parch", inplace=True)
        in_df.drop(columns="SibSp", inplace=True)
        in_df.drop(columns="Fare", inplace=True)
        in_df.drop(columns="Pclass", inplace=True)

        if test:
            return in_df

        return in_df, out_y


def k_data_split(k : int, i : int, data : DataFrame):
    val_ratio = 1.0 / k
    interval = len(data) * val_ratio
    interval = np.floor(interval).astype(np.int16)

    splits = []
    pool = np.array(range(len(data)))
    for j in range(int(1/val_ratio)):
        split = np.random.choice(pool, size=interval, replace=False)
        split = split.tolist()
        splits.append(split)
        pool = pool[np.isin(pool, split, invert=True)]
    #for j in range(len(splits)):
    #    print(len(splits[j]))

    #K-fold Cross-Validation
    val_pool = splits[i]
    train_pool = []
    for j in range(len(splits)):
        if i == j:
            continue
        train_pool.append(splits[j])
    train_pool = np.hstack(train_pool).tolist()

    train_data = data.iloc(axis=0)[train_pool]
    val_data = data.iloc(axis=0)[val_pool]
    return train_data, val_data

def k_cross_val(k : int,
                model_class,
                model_params, data, preprocessor):
    model = model_class(**model_params)

    sum_score = 0.0

    for i in range(k):
        train_data, val_data = k_data_split(k, i, data)

        x_train, y_train = preprocessor.preprocess_dataset(train_data)
        x_val, y_val = preprocessor.preprocess_dataset(val_data)

        model.fit(x_train, np.ravel(y_train))
        val_pred = model.predict(x_val)
        acc = accuracy_score(y_true=y_val, y_pred=val_pred)
        sum_score += acc
    return sum_score / k
