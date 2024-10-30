"""
This module provides a DataPreprocessor class to read JSON data, convert it into a DataFrame, and reshape it to have a row for each id/date and columns for groups 0-3, startingGroups, and a combined array of groups.
"""

import json
import pandas as pd # type: ignore (located in poetry env)
import numpy as np # type: ignore (located in poetry env)
from typing import List


class DataPreprocessor:
    def __init__(self, file_path: str = 'src/data/connection_grouping.json'):
        """
        Initialize the DataPreprocessor with the path to the JSON data file.

        :param file_path: Path to the JSON data file (default: 'src/data/connection_grouping.json')
        """
        self.file_path = file_path
        self.raw_data_df = None
        self.preprocessed_data_df = None
        self.num_groups = 4

    def read_json(self) -> List[dict]:
        """
        Read the JSON data from the file path.

        :return: List of dictionaries containing the data
        """
        with open(self.file_path, 'r') as f:
            return json.load(f)

    def to_dataframe(self, data: List[dict]) -> pd.DataFrame:
        """
        Convert the list of dictionaries to a pandas DataFrame.

        :param data: List of dictionaries containing the data
        :return: pandas DataFrame
        """
        return pd.DataFrame(data)

    def _transform_dict_to_array(self, df_column: str, sub_column: str) -> None:
        """
        Transform the dictionaries in the specified column to arrays.

        :param df_column: Name of the column containing the dictionaries
        :param sub_column: Name of the key in the dictionary
        """
        items_array = self.preprocessed_data_df[df_column].apply(lambda x: [group_data[sub_column] for group_data in x.values()])
        result_df = pd.DataFrame(
            items_array.tolist(),
            columns=[f'group_{i}' for i in range(self.num_groups)]
        )
        self.preprocessed_data_df = pd.concat([self.preprocessed_data_df, result_df], axis=1)

    def _add_group_columns_array(self) -> None:
        """
        Create a new column that contains a 2D array of the group columns.
        """
        group_columns = [f'group_{i}' for i in range(self.num_groups)]
        self.preprocessed_data_df['group_columns_array'] = self.preprocessed_data_df[group_columns].apply(np.array, axis=1)

    def transform(self) -> pd.DataFrame:
        """
        Apply the data preprocessing transformations and return the preprocessed DataFrame.
        """
        self.raw_data_df = self.to_dataframe(self.read_json())
        self.preprocessed_data_df = self.raw_data_df.copy()
        
        self._transform_dict_to_array('groups', 'members')
        self.preprocessed_data_df['startingGroups'] = self.preprocessed_data_df['startingGroups'].apply(np.ravel)
        self._add_group_columns_array()

        return self.preprocessed_data_df

    def split_data(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15) -> tuple:
        """Splits the preprocessed data into training, validation, and testing sets.

        Args:
            train_size: Proportion of data for training.
            val_size: Proportion of data for validation.
            test_size: Proportion of data for testing.

        Returns:
            A tuple containing the training, validation, and testing DataFrames.
        """
        if not (train_size + val_size + test_size) == 1.0:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        np.random.seed(42)
        preprocessed_data = self.transform()
        n = len(preprocessed_data)
        idx = np.random.permutation(n)
        train_idx = idx[:int(train_size * n)]
        val_idx = idx[int(train_size * n):int((train_size + val_size) * n)]
        test_idx = idx[int((train_size + val_size) * n):]

        train_df = preprocessed_data.iloc[train_idx]
        val_df = preprocessed_data.iloc[val_idx]
        test_df = preprocessed_data.iloc[test_idx]

        return train_df, val_df, test_df
