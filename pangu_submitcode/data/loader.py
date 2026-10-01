# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# from typing import List, Optional, Tuple, Union

# from data.loader_utils import WindFarmDataLoader

# class WindFarmDataset(Dataset):
#     """Wind Farm PyTorch Dataset"""

#     def __init__(self,
#                  data_loader: WindFarmDataLoader,
#                  dataset_name: str,
#                  turbine_indices: List[int],
#                  feature_indices: List[int],
#                  target_feature_indices: List[int],  # 新增参数
#                  sequence_length: int = 24,
#                  prediction_horizon: int = 1,
#                  split: str = 'train',
#                  scaler: Optional[dict] = None):
#         """
#         Args:
#             data_loader: WindFarmDataLoader instance
#             dataset_name: Name of dataset to use
#             turbine_indices: List of turbine indices to include
#             feature_indices: List of feature indices to include
#             target_feature_indices: List of feature indices to predict
#             sequence_length: Length of input sequences
#             prediction_horizon: How many steps ahead to predict
#             split: Data split to use ('train', 'val', or 'test')
#             scaler: Dictionary containing mean and std for each feature (for val/test)
#         """
#         self.data = data_loader.get_multiple_turbines_features(
#             dataset_name=dataset_name,
#             turbine_indices=turbine_indices,
#             feature_indices=feature_indices,
#             split=split
#         )
#         self.target_feature_indices = target_feature_indices
#         self.seq_len = sequence_length
#         self.pred_horizon = prediction_horizon

#         # Standardize data
#         if split == 'train':
#             # Calculate mean and std for each feature
#             self.scaler = {
#                 'mean': np.mean(self.data, axis=0),
#                 'std': np.std(self.data, axis=0)
#             }
#         else:
#             # Use provided scaler for val/test
#             assert scaler is not None, "Scaler must be provided for val/test sets"
#             self.scaler = scaler

#         # Apply standardization
#         self.data = (self.data - self.scaler['mean']) / self.scaler['std']

#     def __len__(self) -> int:
#         return len(self.data) - self.seq_len - self.pred_horizon + 1

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
#         """Get input sequence and target value"""
#         # Get input sequence
#         x = self.data[idx:idx + self.seq_len]

#         # Get target value(s) only for specified target features
#         y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_horizon,:, self.target_feature_indices]

#         return torch.FloatTensor(x), torch.FloatTensor(y)


# def create_dataloaders(
#         data_loader: WindFarmDataLoader,
#         dataset_name: str,
#         turbine_indices: List[int],
#         feature_indices: List[int],
#         target_feature_indices: List[int],  # 新增参数
#         batch_size: int = 32,
#         sequence_length: int = 24,
#         prediction_horizon: int = 1,
#         num_workers: int = 0
# ) -> Tuple[DataLoader, DataLoader, DataLoader]:
#     """
#     Create train, validation and test DataLoaders

#     Returns:
#         Tuple of (train_loader, val_loader, test_loader)
#     """
#     # Create train dataset first to get scaler
#     train_dataset = WindFarmDataset(
#         data_loader, dataset_name, turbine_indices, feature_indices,
#         target_feature_indices, sequence_length, prediction_horizon, split='train'
#     )

#     # Create val and test datasets using train scaler
#     val_dataset = WindFarmDataset(
#         data_loader, dataset_name, turbine_indices, feature_indices,
#         target_feature_indices, sequence_length, prediction_horizon, split='val',
#         scaler=train_dataset.scaler
#     )

#     test_dataset = WindFarmDataset(
#         data_loader, dataset_name, turbine_indices, feature_indices,
#         target_feature_indices, sequence_length, prediction_horizon, split='test',
#         scaler=train_dataset.scaler
#     )

#     # Create dataloaders
#     train_loader = DataLoader(
#         train_dataset, batch_size=batch_size, shuffle=True,
#         num_workers=num_workers
#     )

#     val_loader = DataLoader(
#         val_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers
#     )

#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False,
#         num_workers=num_workers
#     )

#     return train_loader, val_loader, test_loader


import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional, Tuple, Union

from pangu_submitcode.data.loader_utils import PanguWeatherDataLoader, WindFarmDataLoader
from pangu_submitcode.data.weather_loader_base import WeatherDataAdapter


class WindFarmDataset(Dataset):
    def __init__(self,
                 wind_data_loader: WindFarmDataLoader,
                 weather_data_loader: Union[PanguWeatherDataLoader, WeatherDataAdapter],  # Support both Pangu and adapter
                 dataset_name: str,
                 turbine_indices: List[int],
                 feature_indices: List[int],
                 target_feature_indices: List[int],
                 sequence_length: int = 36,
                 prediction_horizon: int = 10,
                 split: str = 'train',
                 scaler: Optional[dict] = None,
                 use_weather_data: bool = True
                 ):
        self.use_weather_data = use_weather_data

        # Load wind farm data
        self.wind_data = wind_data_loader.get_multiple_turbines_features(
            dataset_name=dataset_name,
            turbine_indices=turbine_indices,
            feature_indices=feature_indices,
            split=split
        )

        # Load weather data with adapter support (backward compatible)
        if self.use_weather_data:
            # Check if using adapter (new way) or direct loader (old way)
            if isinstance(weather_data_loader, WeatherDataAdapter):
                # New way: using adapter with factory pattern
                self.weather_type = weather_data_loader.get_type()
                weather_data1, weather_data2, metadata = weather_data_loader.get_weather_data(dataset_name, split)

                if self.weather_type == 'pangu' or self.weather_type == 'pangu_random':
                    self.upper_weather = weather_data1
                    self.surface_weather = weather_data2
                    self.fengwu_weather = None
                    self.era5_weather = None
                    print(f"📊 Loaded {self.weather_type.capitalize()} weather data via adapter")
                elif self.weather_type == 'fengwu' or self.weather_type == 'fengwu_random':
                    self.fengwu_weather = weather_data1
                    self.upper_weather = None
                    self.surface_weather = None
                    self.era5_weather = None
                    print(f"📊 Loaded {self.weather_type.capitalize()} weather data via adapter")
                elif self.weather_type == 'era5':
                    self.era5_weather = weather_data1
                    self.fengwu_weather = None
                    self.upper_weather = None
                    self.surface_weather = None
                    print(f"📊 Loaded ERA5 ground truth weather data via adapter")
            else:
                # Old way: direct PanguWeatherDataLoader (backward compatible)
                self.weather_type = 'pangu'
                self.upper_weather, self.surface_weather = weather_data_loader.get_weather_data(dataset_name, split)
                self.fengwu_weather = None
                self.era5_weather = None
                print(f"📊 Loaded Pangu weather data (legacy mode)")
        else:
            self.weather_type = None
            self.upper_weather = None
            self.surface_weather = None
            self.fengwu_weather = None
            self.era5_weather = None

        self.target_feature_indices = target_feature_indices
        self.turbine_indices = turbine_indices
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon

        # Compute scaler for train set, or use provided scaler for val/test
        if split == 'train':
            self.scaler = {
                'wind': {
                    'mean': np.mean(self.wind_data, axis=0),
                    'std': np.std(self.wind_data, axis=0)
                }
            }
            if self.use_weather_data:
                if self.weather_type == 'pangu' or self.weather_type == 'pangu_random':
                    self.scaler.update({
                        'upper': {
                            'mean': np.mean(self.upper_weather, axis=0),
                            'std': np.std(self.upper_weather, axis=0)
                        },
                        'surface': {
                            'mean': np.mean(self.surface_weather, axis=0),
                            'std': np.std(self.surface_weather, axis=0)
                        }
                    })
                elif self.weather_type == 'fengwu' or self.weather_type == 'fengwu_random':
                    self.scaler.update({
                        'fengwu': {
                            'mean': np.mean(self.fengwu_weather, axis=0),
                            'std': np.std(self.fengwu_weather, axis=0)
                        }
                    })
                elif self.weather_type == 'era5':
                    self.scaler.update({
                        'era5': {
                            'mean': np.mean(self.era5_weather, axis=0),
                            'std': np.std(self.era5_weather, axis=0)
                        }
                    })
        else:
            assert scaler is not None, "Scaler must be provided for val/test sets"
            self.scaler = scaler

        # Apply standardization
        self.wind_data = (self.wind_data - self.scaler['wind']['mean']) / self.scaler['wind']['std']
        if self.use_weather_data:
            if self.weather_type == 'pangu' or self.weather_type == 'pangu_random':
                self.upper_weather = (self.upper_weather - self.scaler['upper']['mean']) / self.scaler['upper']['std']
                self.surface_weather = (self.surface_weather - self.scaler['surface']['mean']) / self.scaler['surface']['std']
            elif self.weather_type == 'fengwu' or self.weather_type == 'fengwu_random':
                self.fengwu_weather = (self.fengwu_weather - self.scaler['fengwu']['mean']) / self.scaler['fengwu']['std']
            elif self.weather_type == 'era5':
                self.era5_weather = (self.era5_weather - self.scaler['era5']['mean']) / self.scaler['era5']['std']
            
            
        # print(f"Wind data length: {len(self.wind_data)}")
        # print(f"Upper weather length: {len(self.upper_weather)}")
        # print(f"Surface weather length: {len(self.surface_weather)}")
        # print(f"Sequence length: {self.seq_len}")
        # print(f"Prediction horizon: {self.pred_horizon}")

    def __len__(self):
        return len(self.wind_data) - self.seq_len - self.pred_horizon

    # def __getitem__(self, idx):
    #     wind_x = self.wind_data[idx:idx + self.seq_len]
    #
    #     weather_idx = idx + self.seq_len + self.pred_horizon - 1
    #     upper_weather = self.upper_weather[weather_idx]
    #     surface_weather = self.surface_weather[weather_idx]
    #
    #     y = self.wind_data[idx + self.seq_len:idx + self.seq_len + self.pred_horizon, :, self.target_feature_indices]
    #
    #     return torch.FloatTensor(wind_x), torch.FloatTensor(upper_weather), torch.FloatTensor(surface_weather), torch.FloatTensor(y)

    # def __getitem__(self, idx):
    #     wind_x = self.wind_data[idx:idx + self.seq_len]

    #     if self.use_weather_data:
    #         weather_idx = idx + self.seq_len
    #         upper_weather = self.upper_weather[weather_idx, :(self.pred_horizon // 6)]
    #         surface_weather = self.surface_weather[weather_idx, :(self.pred_horizon // 6)]
    #     else:
    #         upper_weather = None
    #         surface_weather = None

    #     y = self.wind_data[idx + self.seq_len:idx + self.seq_len + self.pred_horizon, :, self.target_feature_indices]

    #     if self.use_weather_data:
    #         return torch.FloatTensor(wind_x), torch.FloatTensor(upper_weather), torch.FloatTensor(
    #             surface_weather), torch.FloatTensor(y)
    #     else:
    #         return torch.FloatTensor(wind_x), torch.FloatTensor(y)


    def __getitem__(self, idx):
        wind_x = self.wind_data[idx:idx + self.seq_len]
        y = self.wind_data[idx + self.seq_len:idx + self.seq_len + self.pred_horizon, :, self.target_feature_indices]

        if self.use_weather_data:
            weather_idx = idx + self.seq_len

            if self.weather_type == 'pangu' or self.weather_type == 'pangu_random':
                # Pangu mode (real or random): return (wind_x, upper, surface, y)
                weather_idx = min(weather_idx, len(self.upper_weather) - 1)
                upper_weather = self.upper_weather[weather_idx, :(self.pred_horizon // 6)]
                surface_weather = self.surface_weather[weather_idx, :(self.pred_horizon // 6)]
                return (torch.FloatTensor(wind_x),
                       torch.FloatTensor(upper_weather),
                       torch.FloatTensor(surface_weather),
                       torch.FloatTensor(y))

            elif self.weather_type == 'fengwu' or self.weather_type == 'fengwu_random':
                # Fengwu mode (real or random): return (wind_x, fengwu, y)
                weather_idx = min(weather_idx, len(self.fengwu_weather) - 1)
                fengwu_weather = self.fengwu_weather[weather_idx]  # Shape: (69, 11, 11)
                return (torch.FloatTensor(wind_x),
                       torch.FloatTensor(fengwu_weather),
                       torch.FloatTensor(y))

            elif self.weather_type == 'era5':
                # ERA5 mode: return (wind_x, era5, y)
                weather_idx = min(weather_idx, len(self.era5_weather) - 1)
                era5_weather = self.era5_weather[weather_idx]  # Shape: (69, 11, 11)
                return (torch.FloatTensor(wind_x),
                       torch.FloatTensor(era5_weather),
                       torch.FloatTensor(y))
        else:
            # No weather data: return (wind_x, y)
            return torch.FloatTensor(wind_x), torch.FloatTensor(y)



class WindFarmDataset_onlytest(Dataset):
    """
    专门用于获取ground truth数据的Dataset类
    支持直接访问完整数据，避免通过DataLoader循环的开销
    """
    def __init__(self,
                 wind_data_loader: WindFarmDataLoader,
                 dataset_name: str,
                 turbine_indices: List[int],
                 feature_indices: List[int],
                 target_feature_indices: List[int],
                 sequence_length: int = 36,
                 prediction_horizon: int = 10,
                 split: str = 'test',
                 scaler: Optional[dict] = None,
                 use_weather_data: bool = False
                 ):
        self.use_weather_data = use_weather_data

        # 获取原始风场数据
        self.original_wind_data = wind_data_loader.get_multiple_turbines_features(
            dataset_name=dataset_name,
            turbine_indices=turbine_indices,
            feature_indices=feature_indices,
            split=split
        )

        self.target_feature_indices = target_feature_indices
        self.turbine_indices = turbine_indices
        self.seq_len = sequence_length
        self.pred_horizon = prediction_horizon
        self.split = split

        # 处理标准化
        if split == 'train':
            # 如果是训练集，计算标准化参数
            self.scaler = {
                'wind': {
                    'mean': np.mean(self.original_wind_data, axis=0),
                    'std': np.std(self.original_wind_data, axis=0)
                }
            }
            # 应用标准化
            self.wind_data = (self.original_wind_data - self.scaler['wind']['mean']) / self.scaler['wind']['std']
        else:
            # 如果是验证集或测试集，使用提供的标准化参数
            if scaler is not None:
                self.scaler = scaler
                # 应用标准化
                self.wind_data = (self.original_wind_data - self.scaler['wind']['mean']) / self.scaler['wind']['std']
            else:
                # 如果没有提供scaler，使用原始数据
                self.scaler = None
                self.wind_data = self.original_wind_data
                
    def __len__(self):
        """返回可以生成的样本数量"""
        return len(self.original_wind_data) - self.seq_len - self.pred_horizon + 1

    def __getitem__(self, idx):
        """
        根据索引返回对应的ground truth数据
        返回该时间步对应的目标值（用于与预测结果对比）
        """
        # 计算目标时间步的范围
        start_idx = idx + self.seq_len
        end_idx = start_idx + self.pred_horizon
        
        # 获取原始数据和标准化数据的目标值
        y_original = self.original_wind_data[start_idx:end_idx, :, self.target_feature_indices]
        
        if self.scaler is not None:
            y_normalized = self.wind_data[start_idx:end_idx, :, self.target_feature_indices]
        else:
            y_normalized = y_original
        
        return torch.FloatTensor(y_original), torch.FloatTensor(y_normalized)

    def get_full_original_data(self):
        """
        获取完整的原始数据（未标准化）
        主要用于直接保存ground truth
        """
        if isinstance(self.target_feature_indices, list):
            return self.original_wind_data[:, :, self.target_feature_indices]
        else:
            return self.original_wind_data[:, :, [self.target_feature_indices]]
    
    def get_full_normalized_data(self):
        """
        获取完整的标准化数据
        用于与模型预测结果对比
        """
        if isinstance(self.target_feature_indices, list):
            return self.wind_data[:, :, self.target_feature_indices]
        else:
            return self.wind_data[:, :, [self.target_feature_indices]]
    
    def get_target_sequences(self):
        """
        获取所有目标序列（与模型预测对应的时间步）
        返回形状：(num_sequences, pred_horizon, num_turbines, num_target_features)
        """
        num_sequences = len(self)
        target_sequences_original = []
        target_sequences_normalized = []
        
        for i in range(num_sequences):
            start_idx = i + self.seq_len
            end_idx = start_idx + self.pred_horizon
            
            y_orig = self.original_wind_data[start_idx:end_idx, :, self.target_feature_indices]
            target_sequences_original.append(y_orig)
            
            if self.scaler is not None:
                y_norm = self.wind_data[start_idx:end_idx, :, self.target_feature_indices]
                target_sequences_normalized.append(y_norm)
            else:
                target_sequences_normalized.append(y_orig)
        
        return np.array(target_sequences_original), np.array(target_sequences_normalized)
    
    def get_reconstructed_timeline(self, use_normalized=True):
        """
        重建完整的时间线（处理重叠预测）
        与visualize_and_save_predictions函数中的逻辑保持一致
        
        Args:
            use_normalized: 是否使用标准化数据
            
        Returns:
            reconstructed_data: 重建后的时间序列
        """
        num_sequences = len(self)
        total_timesteps = self.seq_len + (num_sequences - 1) + self.pred_horizon
        
        # 选择使用哪种数据
        source_data = self.wind_data if use_normalized else self.original_wind_data
        
        # 初始化重建数组
        reconstructed = np.full(total_timesteps, np.nan)
        counts = np.zeros(total_timesteps)
        
        # 处理目标特征
        if isinstance(self.target_feature_indices, list):
            if len(self.target_feature_indices) == 1:
                target_data = source_data[:, :, self.target_feature_indices[0]]
            else:
                target_data = source_data[:, :, self.target_feature_indices[0]]  # 取第一个特征
        else:
            target_data = source_data[:, :, self.target_feature_indices]
        
        # 如果有多个turbine，取第一个
        if target_data.ndim > 1:
            target_data = target_data[:, 0]  # 取第一个turbine
        
        # 重建时间序列
        for i in range(num_sequences):
            start_idx = i + self.seq_len
            end_idx = start_idx + self.pred_horizon
            
            # 累加数据
            reconstructed[start_idx:end_idx] = np.nansum([
                reconstructed[start_idx:end_idx], 
                target_data[start_idx:end_idx]
            ], axis=0)
            counts[start_idx:end_idx] += 1
        
        # 计算平均值
        with np.errstate(divide='ignore', invalid='ignore'):
            reconstructed /= counts
        
        # 移除开头和结尾的NaN值
        valid_indices = np.where(~np.isnan(reconstructed))[0]
        if len(valid_indices) > 0:
            first_valid = valid_indices[0]
            last_valid = valid_indices[-1]
            reconstructed = reconstructed[first_valid:last_valid + 1]
        
        return reconstructed




from torch.utils.data import DataLoader


def create_dataloaders(
        wind_data_loader: WindFarmDataLoader,
        weather_data_loader: Union[PanguWeatherDataLoader, WeatherDataAdapter],  # Support both types
        dataset_name: str,
        turbine_indices: List[int],
        feature_indices: List[int],
        target_feature_indices: List[int],
        batch_size: int = 32,
        sequence_length: int = 36,
        prediction_horizon: int = 10,
        num_workers: int = 0,
        use_weather_data: bool = True


) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = WindFarmDataset(
        wind_data_loader, weather_data_loader, dataset_name, turbine_indices, feature_indices,
        target_feature_indices, sequence_length, prediction_horizon, split='train', use_weather_data=use_weather_data,
    )

    val_dataset = WindFarmDataset(
        wind_data_loader, weather_data_loader, dataset_name, turbine_indices, feature_indices,
        target_feature_indices, sequence_length, prediction_horizon, split='val',
        scaler=train_dataset.scaler, use_weather_data=use_weather_data
    )

    test_dataset = WindFarmDataset(
        wind_data_loader, weather_data_loader, dataset_name, turbine_indices, feature_indices,
        target_feature_indices, sequence_length, prediction_horizon, split='test',
        scaler=train_dataset.scaler, use_weather_data=use_weather_data
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


# 使用示例:
def example_usage_dataloader():
    # 初始化数据加载器
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed")
    loader = WindFarmDataLoader(data_dir=PROCESSED_DIR)

    # 设置参数
    dataset_name = 'wf1'
    turbine_indices = [0]  # 使用第一个风机
    feature_indices = [0, 1]  # 使用前两个特征作为输入
    target_feature_indices = [0]  # 只预测第一个特征

    # 创建dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        loader,
        dataset_name=dataset_name,
        turbine_indices=turbine_indices,
        feature_indices=feature_indices,
        target_feature_indices=target_feature_indices,
        batch_size=32,
        sequence_length=36,
        prediction_horizon=10
    )

    # 打印数据集信息
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    # 获取一个batch的数据示例
    x_batch, y_batch = next(iter(train_loader))
    print(f"\nInput batch shape: {x_batch.shape}")  # 应该是 [batch_size, seq_len, num_features]
    print(f"Target batch shape: {y_batch.shape}")  # 应该是 [batch_size, pred_horizon, num_target_features]

    # 打印标准化参数
    print("\nStandardization parameters:")
    print(f"Mean: {train_loader.dataset.scaler['mean']}")
    print(f"Std: {train_loader.dataset.scaler['std']}")


if __name__ == "__main__":
    example_usage_dataloader()
