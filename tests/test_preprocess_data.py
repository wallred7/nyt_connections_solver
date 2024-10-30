import unittest
import numpy as np # type: ignore
from src.data.preprocess_data import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        self.preprocessed_data = self.preprocessor.transform()

    def test_transform(self):
        self.assertIn('id', self.preprocessed_data.columns)
        for i in range(self.preprocessor.num_groups):
            self.assertIn(f'group_{i}', self.preprocessed_data.columns)
        self.assertIn('startingGroups', self.preprocessed_data.columns)
        self.assertIn('group_columns_array', self.preprocessed_data.columns)

    def test_group_columns_array(self):
        expected_group_array = self.preprocessed_data[['group_0', 'group_1', 'group_2', 'group_3']].apply(np.array, axis=1)
        self.assertTrue(self.preprocessed_data['group_columns_array'].equals(expected_group_array))

if __name__ == '__main__':
    unittest.main()
