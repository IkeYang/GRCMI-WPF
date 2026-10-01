import os
from datetime import datetime, timedelta
import cdsapi
import numpy as np
import netCDF4 as nc
import time
import logging
from functools import wraps

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_download.log'),
        logging.StreamHandler()
    ]
)

def retry_on_error(max_retries=3, delay=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"Maximum retries ({max_retries}) reached. Error: {str(e)}")
                        raise
                    logging.warning(f"Error occurred: {str(e)}. Retrying in {delay} seconds... (Attempt {retries}/{max_retries})")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

# Path to the folder containing time-based subfolders
# folder_path = 'D:/WY/data_pangu/forecasts'
folder_path = os.path.join(os.getcwd(), 'forecasts')

# Function to get a list of existing datetime objects from folder names
def get_existing_times(folder_path):
    existing_times = []
    for folder_name in os.listdir(folder_path):
        try:
            folder_path_full = os.path.join(folder_path, folder_name)
            # 检查文件夹中是否包含所有必需的文件，且文件大小不为0
            required_files = ['surface.nc', 'upper.nc', 'input_surface.npy', 'input_upper.npy']
            if all(os.path.exists(os.path.join(folder_path_full, f)) and 
                  os.path.getsize(os.path.join(folder_path_full, f)) > 0 
                  for f in required_files):
                existing_time = datetime.strptime(folder_name, '%Y-%m-%d-%H-%M')
                existing_times.append(existing_time)
            else:
                # 如果文件不完整，删除这个文件夹，以便重新下载
                logging.warning(f"发现不完整数据文件夹: {folder_name}，将删除重新下载")
                for f in os.listdir(folder_path_full):
                    try:
                        os.remove(os.path.join(folder_path_full, f))
                    except Exception as e:
                        logging.error(f"删除文件失败: {str(e)}")
                try:
                    os.rmdir(folder_path_full)
                except Exception as e:
                    logging.error(f"删除文件夹失败: {str(e)}")
        except ValueError:
            continue
    return existing_times

# Define the time range and interval
start_time = datetime(2019, 6, 1, 0, 0)
end_time = datetime(2020, 3, 1, 0, 0)
interval = timedelta(hours=1)

# Get list of existing times
existing_times = set(get_existing_times(folder_path))

# Identify missing times
current_time = start_time
missing_times = []

while current_time <= end_time:
    if current_time not in existing_times:
        missing_times.append(current_time)
    current_time += interval

# Download missing data
c = cdsapi.Client()
surface_variables = ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature']
upper_variables = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind']
area = [90, 0, -90, 359.75]
pressure_levels = ['1000', '925', '850', '700', '600', '500', '400', '300', '250', '200', '150', '100', '50']
print('miss_time:', missing_times)

while True:  # 无限循环
    try:
        # 每次循环重新获取缺失时间列表
        existing_times = set(get_existing_times(folder_path))
        missing_times = []
        current_time = start_time
        while current_time <= end_time:
            if current_time not in existing_times:
                missing_times.append(current_time)
            current_time += interval
            
        if not missing_times:  # 如果没有缺失数据，退出循环
            logging.info("所有数据已完成下载和处理")
            break
            
        logging.info(f"开始处理缺失数据，共{len(missing_times)}个时间点")
        
        for date_time in missing_times:
            print('current_time:', date_time)
            forecast_dir = os.path.join(folder_path, date_time.strftime("%Y-%m-%d-%H-%M"))
            os.makedirs(forecast_dir, exist_ok=True)

            # Download the surface data
            logging.info(f"Downloading surface data for {date_time}")
            c.retrieve('reanalysis-era5-single-levels', {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': surface_variables,
                'date': date_time.strftime("%Y-%m-%d"),
                'time': date_time.strftime("%H:%M"),
                'area': area,
            }, os.path.join(forecast_dir, 'surface.nc'))

            # Download the upper air data
            logging.info(f"Downloading upper air data for {date_time}")
            c.retrieve('reanalysis-era5-pressure-levels', {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': upper_variables,
                'pressure_level': pressure_levels,
                'date': date_time.strftime("%Y-%m-%d"),
                'time': date_time.strftime("%H:%M"),
                'area': area,
            }, os.path.join(forecast_dir, 'upper.nc'))

            # Convert the surface data to npy
            logging.info(f"Processing surface data for {date_time}")
            surface_data = np.zeros((4, 721, 1440), dtype=np.float32)
            with nc.Dataset(os.path.join(forecast_dir, 'surface.nc')) as nc_file:
                surface_data[0] = nc_file.variables['msl'][:].astype(np.float32)
                surface_data[1] = nc_file.variables['u10'][:].astype(np.float32)
                surface_data[2] = nc_file.variables['v10'][:].astype(np.float32)
                surface_data[3] = nc_file.variables['t2m'][:].astype(np.float32)
            np.save(os.path.join(forecast_dir, 'input_surface.npy'), surface_data)

            # Convert the upper air data to npy
            logging.info(f"Processing upper air data for {date_time}")
            upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32)
            with nc.Dataset(os.path.join(forecast_dir, 'upper.nc')) as nc_file:
                upper_data[0] = (nc_file.variables['z'][:]).astype(np.float32)
                upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
                upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
                upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
                upper_data[4] = nc_file.variables['v'][:].astype(np.float32)
            np.save(os.path.join(forecast_dir, 'input_upper.npy'), upper_data)

            # 在每个时间点处理完成后，验证文件完整性
            forecast_dir = os.path.join(folder_path, date_time.strftime("%Y-%m-%d-%H-%M"))
            required_files = ['surface.nc', 'upper.nc', 'input_surface.npy', 'input_upper.npy']
            if not all(os.path.exists(os.path.join(forecast_dir, f)) and 
                      os.path.getsize(os.path.join(forecast_dir, f)) > 0 
                      for f in required_files):
                raise Exception(f"数据处理未完全完成: {date_time}")

            logging.info(f"Successfully processed data for {date_time}")

    except Exception as e:
        logging.error(f"发生错误: {str(e)}")
        logging.info("等待60秒后重试...")
        time.sleep(60)
