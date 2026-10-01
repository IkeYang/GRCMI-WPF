import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime

# 添加一个自定义的 pickle 加载函数来处理兼容性问题
def safe_pickle_load(file_path: str) -> Dict:
    """
    Safely load pickle file with pandas timestamp compatibility
    """
    try:
        # 尝试直接加载
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except TypeError as e:
        if "takes exactly 3 positional arguments (4 given)" in str(e):
            # 如果出现 timestamp 错误，使用更安全的加载方式
            pd.options.mode.use_inf_as_na = True
            with open(file_path, 'rb') as f:
                try:
                    # 使用 latin1 编码尝试加载
                    return pickle.load(f, encoding='latin1')
                except:
                    # 如果还是失败，尝试使用 bytes 编码
                    return pickle.load(f, encoding='bytes')
        else:
            raise e

class PanguWeatherDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_upper_layer(self, split: str) -> np.ndarray:
        # 加载上层数据
        # 实现数据的重塑：(时间, 5, 13, 11, 11) → (时间, 5*13*11*11)
        pass

    def load_surface_layer(self, split: str) -> np.ndarray:
        # 加载表层数据
        # 实现数据的重塑：(时间, 4, 11, 11) → (时间, 4*11*11)
        pass

    def get_weather_data(self, split: str) -> Tuple[np.ndarray, np.ndarray]:
        upper = self.load_upper_layer(split)
        surface = self.load_surface_layer(split)
        return upper, surface


class WindFarmDataLoader:
    def __init__(self, data_dir: str, test_size: float = 0.2, val_size: float = 0.2, train_size_fraction: float = 0.6, val_size_fraction: float = 0.2, random_state: int = 42):
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.train_size_fraction = train_size_fraction
        self.val_size_fraction = (1-self.train_size_fraction) /2
        self.random_state = random_state

        # Store loaded data
        self.loaded_data = {}

        # Feature names will be loaded from the processed data files
        self.feature_names = None

        # 修改dataset_specs，将La Haute Borne分为两个数据集
        self.dataset_specs = {
            'wf1': {
                'description': 'Maitas Phase 1 (33 turbines)',
                'n_turbines': 33,
                'format': 'pickle',
                'filename': 'wf1_data_processed.pkl'
            },
            'wf2': {
                'description': 'Maitas Phase 2 (33 turbines)',
                'n_turbines': 33,
                'format': 'pickle',
                'filename': 'wf2_data_processed.pkl'
            },
            'wf3': {
                'description': 'Tieleketi (25 turbines)',
                'n_turbines': 25,
                'format': 'pickle',
                'filename': 'wf3_data_processed.pkl'
            },
            'lhb_2013_2016': {
                'description': 'La Haute Borne dataset 2013-2016',
                'format': 'pickle',
                'filename': 'la-haute-borne-data-2013-2016_processed.pkl'
            },
            'lhb_2017_2020': {
                'description': 'La Haute Borne dataset 2017-2020',
                'format': 'pickle',
                # 'filename': 'la-haute-borne-data-2017-2020_processed.pkl'
                'filename': 'la-haute-borne-data-2017-2020_processed_2017_01_01_to_2017_12_31.pkl'
            },
            'beslona': {
                'files': ['Beslona17Train', 'Beslona17Val', 'Beslona17Test'],
                'description': 'Beslona dataset',
                'format': 'pickle'
            }
        }

    def load_train_data(self, dataset_name: str, turbine_indices: List[int], feature_indices: List[int]) -> np.ndarray:
        """
        Load training data for specified turbines and features

        Args:
            dataset_name: Name of the dataset
            turbine_indices: List of turbine indices to select
            feature_indices: List of feature indices to select

        Returns:
            Training data array of shape (timesteps, n_selected_turbines, n_selected_features)
        """
        return self.get_multiple_turbines_features(
            dataset_name=dataset_name,
            turbine_indices=turbine_indices,
            feature_indices=feature_indices,
            split='train'
        )

    def load_val_data(self, dataset_name: str, turbine_indices: List[int], feature_indices: List[int]) -> np.ndarray:
        """
        Load validation data for specified turbines and features
        """
        return self.get_multiple_turbines_features(
            dataset_name=dataset_name,
            turbine_indices=turbine_indices,
            feature_indices=feature_indices,
            split='val'
        )

    def load_test_data(self, dataset_name: str, turbine_indices: List[int], feature_indices: List[int]) -> np.ndarray:
        """
        Load test data for specified turbines and features
        """
        return self.get_multiple_turbines_features(
            dataset_name=dataset_name,
            turbine_indices=turbine_indices,
            feature_indices=feature_indices,
            split='test'
        )
    def read_pickle_data(self, filename: str) -> Dict:
        """Read pickle data from file with compatibility handling"""
        full_path = os.path.join(self.data_dir, filename)
        return safe_pickle_load(full_path)

    def read_csv_data(self, filename: str) -> pd.DataFrame:
        """Read CSV data from file"""
        full_path = os.path.join(self.data_dir, filename)
        return pd.read_csv(full_path)

    def load_dataset(self, dataset_name: str) -> Dict:
        """
        Load a specific dataset by name

        Args:
            dataset_name: Name of the dataset to load ('wf1', 'wf2', 'wf3', 'la_haute_borne', 'beslona')

        Returns:
            Dictionary containing the loaded dataset
        """
        if dataset_name in self.loaded_data:
            return self.loaded_data[dataset_name]

        if dataset_name not in self.dataset_specs:
            raise ValueError(f"Unknown dataset: {dataset_name}")


        if dataset_name in ['wf1', 'wf2', 'wf3']:
            # 保持原有的wind farm数据处理逻辑
            raw_data = self.read_pickle_data(self.dataset_specs[dataset_name]['filename'])

            if self.feature_names is None:
                self.feature_names = raw_data['feature_names']

            data = raw_data['data']
            train, val, test = self._split_wind_farm_data(data)

            dataset = {
                'train': train,
                'val': val,
                'test': test,
                'full': data,
                'timestamps': raw_data['timestamps'],
                'turbine_ids': raw_data['turbine_ids']
            }

        elif dataset_name in ['lhb_2013_2016', 'lhb_2017_2020']:
            # La Haute Borne数据处理逻辑
            raw_data = self.read_pickle_data(self.dataset_specs[dataset_name]['filename'])
            data = raw_data['data']
            train, val, test = self._split_wind_farm_data(data)

            dataset = {
                'train': train,
                'val': val,
                'test': test,
                'full': data,
                'timestamps': raw_data['timestamps'],
                'turbine_ids': raw_data['turbine_ids'],
                'feature_names': raw_data['feature_names']
            }

        elif dataset_name == 'beslona':
            # 保持原有的beslona数据处理逻辑
            train = self.read_pickle_data(self.dataset_specs[dataset_name]['files'][0])
            val = self.read_pickle_data(self.dataset_specs[dataset_name]['files'][1])
            test = self.read_pickle_data(self.dataset_specs[dataset_name]['files'][2])
            dataset = {
                'train': train,
                'val': val,
                'test': test
            }

        self.loaded_data[dataset_name] = dataset
        return dataset


    def get_feature_names(self, dataset_name: Optional[str] = None) -> List[str]:
        """
        Get feature names for specified dataset
        """
        if dataset_name is None or dataset_name in ['wf1', 'wf2', 'wf3']:
            if self.feature_names is None:
                self.load_dataset('wf1')
            return self.feature_names
        elif dataset_name in ['lhb_2013_2016', 'lhb_2017_2020']:
            dataset = self.load_dataset(dataset_name)
            return dataset['feature_names']
        else:
            raise ValueError(f"Feature names not available for dataset: {dataset_name}")

    def get_turbine_data(self, dataset_name: str, turbine_idx: int,
                         split: str = 'full') -> np.ndarray:
        """Get data for a specific turbine"""
        if dataset_name not in ['wf1', 'wf2', 'wf3', 'lhb_2013_2016', 'lhb_2017_2020']:
            raise ValueError("Turbine-specific data not available for this dataset")

        dataset = self.load_dataset(dataset_name)
        if split not in dataset:
            raise ValueError(f"Invalid split '{split}' for {dataset_name}")

        data = dataset[split]
        return data[:, turbine_idx, :]

    def get_turbine_features(self, dataset_name: str, turbine_idx: int,
                             feature_indices: List[int], split: str = 'full') -> np.ndarray:
        """
        Get specific features for a specific turbine from a dataset

        Args:
            dataset_name: Name of the dataset
            turbine_idx: Index of the turbine
            feature_indices: List of feature indices to select
            split: Data split to use ('full', 'train', 'val', 'test')

        Returns:
            Numpy array containing selected features for the specified turbine
        """
        # Get data for the specified turbine
        turbine_data = self.get_turbine_data(dataset_name, turbine_idx, split)

        # Select specified features
        return turbine_data[:, feature_indices]

    def get_multiple_turbines_features(self, dataset_name: str, turbine_indices: List[int],
                                       feature_indices: List[int], split: str = 'full') -> np.ndarray:
        """
        Get specific features for multiple turbines from a dataset

        Args:
            dataset_name: Name of the dataset
            turbine_indices: List of turbine indices
            feature_indices: List of feature indices to select
            split: Data split to use ('full', 'train', 'val', 'test')

        Returns:
            Numpy array containing selected features for specified turbines
            Shape: (timesteps, n_selected_turbines, n_selected_features)
        """
        dataset = self.load_dataset(dataset_name)
        data = dataset[split]

        
        ###
        
        print('wind_data timestamps', dataset['timestamps'])
        # Select specified turbines and features
        return data[:, turbine_indices][:, :, feature_indices]

    def get_dataset_info(self, dataset_name: Optional[str] = None) -> Dict:
        """
        Get information about datasets

        Args:
            dataset_name: Specific dataset to get info about (optional)

        Returns:
            Dictionary containing dataset information
        """
        if dataset_name is not None:
            if dataset_name not in self.dataset_specs:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            return self.dataset_specs[dataset_name]
        return self.dataset_specs

    def get_timestamps(self, dataset_name: str) -> np.ndarray:
        """
        Get timestamps for a specific dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            Numpy array of timestamps
        """
        if dataset_name not in ['wf1', 'wf2', 'wf3']:
            raise ValueError("Timestamps only available for wind farm datasets")

        dataset = self.load_dataset(dataset_name)
        return dataset['timestamps']

    def get_turbine_ids(self, dataset_name: str) -> List[str]:
        """
        Get turbine IDs for a specific dataset

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of turbine IDs
        """
        if dataset_name not in ['wf1', 'wf2', 'wf3']:
            raise ValueError("Turbine IDs only available for wind farm datasets")

        dataset = self.load_dataset(dataset_name)
        return dataset['turbine_ids']

    # def _split_wind_farm_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Split wind farm data into train, validation, and test sets using temporal splitting

    #     Args:
    #         data: Input data of shape (timesteps, n_turbines, n_features)

    #     Returns:
    #         Tuple of (train, validation, test) datasets
    #     """
    #     total_timesteps = data.shape[0]

    #     # Calculate split indices
    #     test_size = int(total_timesteps * self.test_size)
    #     val_size = int((total_timesteps - test_size) * self.val_size)
    #     train_size = total_timesteps - test_size - val_size

    #     # Split the data temporally
    #     train = data[:train_size]
    #     val = data[train_size:train_size + val_size]
    #     test = data[train_size + val_size:]

    #     return train, val, test
    def _split_wind_farm_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_timesteps = data.shape[0]

        # Define absolute split proportions
        train_size = int(self.train_size_fraction * total_timesteps)
        val_size = int(self.val_size_fraction * total_timesteps)
        test_size = total_timesteps - train_size - val_size

        # Perform the split
        train = data[:train_size]
        val = data[train_size:train_size + val_size]
        test = data[train_size + val_size:]

        return train, val, test



import numpy as np
import os
from typing import Tuple, Dict

class PanguWeatherDataLoader:
    def __init__(self, data_dir: str, test_size: float = 0.2, val_size: float = 0.2):
        self.data_dir = data_dir
        self.test_size = test_size
        self.val_size = val_size
        self.loaded_data = {}
        
    def load_dataset(self, dataset_name: str) -> Dict[str, Dict[str, np.ndarray]]:
        if dataset_name in self.loaded_data:
            return self.loaded_data[dataset_name]

        upper_file = f'pangu4{dataset_name}_upper.npy'
        surface_file = f'pangu4{dataset_name}_surface.npy'

        try:
            # upper_data = np.load(os.path.join(self.data_dir, upper_file))
            # surface_data = np.load(os.path.join(self.data_dir, surface_file))

            upper_data = np.load(os.path.join(self.data_dir, upper_file), allow_pickle=True).item()['data']
            surface_data = np.load(os.path.join(self.data_dir, surface_file), allow_pickle=True).item()['data']

            ###
            upper_data_time = np.load(os.path.join(self.data_dir, upper_file), allow_pickle=True).item()['time_labels']
            surface_data_time = np.load(os.path.join(self.data_dir, surface_file), allow_pickle=True).item()['time_labels']
            print(f"Upper data shape: {upper_data.shape}")
            print(f"Surface data shape: {surface_data.shape}")
            print(f"Upper data time shape: {upper_data_time.shape}")
            print(f"Surface data time shape: {surface_data_time.shape}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Pangu data files not found for dataset: {dataset_name}")

        # Split data
        upper_train, upper_val, upper_test = self._split_data(upper_data)
        surface_train, surface_val, surface_test = self._split_data(surface_data)

        dataset = {
            'upper': {
                'train': upper_train,
                'val': upper_val,
                'test': upper_test,
                'full': upper_data
            },
            'surface': {
                'train': surface_train,
                'val': surface_val,
                'test': surface_test,
                'full': surface_data
            }
        }

        self.loaded_data[dataset_name] = dataset
        return dataset

    def _split_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_length = len(data)
        test_split = int((1 - self.test_size) * total_length)
        val_split = int((1 - self.test_size - self.val_size) * total_length)

        train = data[:val_split]
        val = data[val_split:test_split]
        test = data[test_split:]

        return train, val, test

    def get_weather_data(self, dataset_name: str, split: str = 'full') -> Tuple[np.ndarray, np.ndarray]:
        dataset = self.load_dataset(dataset_name)
        return dataset['upper'][split], dataset['surface'][split]




def test_wind_farm_data_loader():
     # 获取脚本所在目录的绝对路径
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 构建到 processed 目录的路径
    PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")

    # 初始化数据加载器时使用这个路径
    loader = WindFarmDataLoader(data_dir=PROCESSED_DIR)

    # Test loading specific dataset
    wf1_data = loader.load_dataset('wf1')
    print("\nWF1 Data Shapes:")
    print(f"Full: {wf1_data['full'].shape}")
    print(f"Train: {wf1_data['train'].shape}")
    print(f"Val: {wf1_data['val'].shape}")
    print(f"Test: {wf1_data['test'].shape}")

    # Test getting specific turbine data
    turbine_data = loader.get_turbine_data('wf1', turbine_idx=0, split='train')
    print(f"\nTurbine 0 Training Data Shape: {turbine_data.shape}")
    print(f"Feature names: {loader.get_feature_names()}")

    # Test getting timestamps and turbine IDs
    timestamps = loader.get_timestamps('wf1')
    turbine_ids = loader.get_turbine_ids('wf1')
    print(f"\nTimestamps shape: {timestamps.shape}")
    print(f"First timestamp: {timestamps[0]}")
    print(f"Last timestamp: {timestamps[-1]}")
    print(f"Turbine IDs: {turbine_ids}")

    # Print dataset info
    print("\nDataset Information:")
    for dataset_name in ['wf1', 'wf2', 'wf3']:
        info = loader.get_dataset_info(dataset_name)
        print(f"\n{dataset_name}:")
        print(f"Description: {info['description']}")
        print(f"Number of turbines: {info['n_turbines']}")

    print("\nTest completed successfully!")
    return loader


def example_usage():
    # 初始化数据加载器
    # 获取脚本所在目录的绝对路径
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 构建到 processed 目录的路径
    PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")

    # 初始化数据加载器时使用这个路径
    loader = WindFarmDataLoader(data_dir=PROCESSED_DIR)

    # 1. 单个风机的特定特征
    # 例如：获取风机0的功率和风速数据（假设它们是前两个特征）
    turbine_idx = 0
    feature_indices = [0, 1]  # 功率和风速的索引

    # 获取数据
    data = loader.get_turbine_features('wf3', turbine_idx, feature_indices)
    print("\n单个风机特定特征数据:")
    print(f"Shape: {data.shape}")
    print(f"选择的特征: {[loader.get_feature_names()[i] for i in feature_indices]}")

    # 2. 多个风机的特定特征
    # 例如：获取前3个风机的功率和风速数据
    turbine_indices = [0, 1, 2]
    feature_indices = [0, 1]

    # 获取数据
    multi_data = loader.get_multiple_turbines_features('wf1', turbine_indices, feature_indices)
    print("\n多个风机特定特征数据:")
    print(f"Shape: {multi_data.shape}")
    print(f"选择的风机: {[loader.get_turbine_ids('wf1')[i] for i in turbine_indices]}")
    print(f"选择的特征: {[loader.get_feature_names()[i] for i in feature_indices]}")

    # 3. 展示数据示例
    print("\n数据示例 (前5个时间点):")
    timestamps = loader.get_timestamps('wf1')[:5]
    for i, ts in enumerate(timestamps):
        print(f"\n时间点 {ts}:")
        for t_idx, turbine_id in enumerate(turbine_indices):
            print(f"风机 {loader.get_turbine_ids('wf1')[turbine_id]}:")
            for f_idx, feature in enumerate(feature_indices):
                feature_name = loader.get_feature_names()[feature]
                value = multi_data[i, t_idx, f_idx]
                print(f"  {feature_name}: {value:.2f}")



if __name__ == "__main__":
    # test_la_haute_borne()
    #
    # loader = test_wind_farm_data_loader()
    example_usage()


    # def test_la_haute_borne():
    #     # 获取脚本所在目录的绝对路径
    #     CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    #     # 构建到 processed 目录的路径
    #     PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")
    #
    #     # 初始化数据加载器时使用这个路径
    #     loader = WindFarmDataLoader(data_dir=PROCESSED_DIR)
    #
    #     # # 测试2013-2016数据集
    #     # print("\nTesting La Haute Borne 2013-2016 dataset:")
    #     lhb_2013_2016 = loader.load_dataset('lhb_2013_2016')
    #     # print(f"Data shape: {lhb_2013_2016['full'].shape}")
    #     # print(f"Number of turbines: {len(lhb_2013_2016['turbine_ids'])}")
    #     # print(f"Time range: {lhb_2013_2016['timestamps'][0]} to {lhb_2013_2016['timestamps'][-1]}")
    #     #
    #     # # 测试2017-2020数据集
    #     # print("\nTesting La Haute Borne 2017-2020 dataset:")
    #     # lhb_2017_2020 = loader.load_dataset('lhb_2017_2020')
    #     # print(f"Data shape: {lhb_2017_2020['full'].shape}")
    #     # print(f"Number of turbines: {len(lhb_2017_2020['turbine_ids'])}")
    #     # print(f"Time range: {lhb_2017_2020['timestamps'][0]} to {lhb_2017_2020['timestamps'][-1]}")
    #
    #     # 获取特定风机和特征的数据
    #     turbine_idx = 0
    #     feature_indices = [0, 1]  # 假设我们要获取前两个特征
    #
    #     # 从2013-2016数据集获取数据
    #     data_2013_2016 = loader.get_turbine_features('lhb_2013_2016', turbine_idx, feature_indices)
    #     print(f"\nSelected turbine features shape (2013-2016): {data_2013_2016.shape}")
    #
    #     # 从2017-2020数据集获取数据
    #     data_2017_2020 = loader.get_turbine_features('lhb_2017_2020', turbine_idx, feature_indices)
    #     print(f"Selected turbine features shape (2017-2020): {data_2017_2020.shape}")
