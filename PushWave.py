# -*- coding: utf-8 -*-
"""
推波助澜择时模型（Hull移动平均线版本）
输出：每日持仓情况Excel文件 + 涨停跌停股票列表 + HTML格式策略报告
"""

import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
import warnings
import os
import json
from typing import Dict, List, Optional, Tuple, Set
import warnings
import math
import base64
import io
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import a_passwards as pw
import email_sender_v2
warnings.filterwarnings('ignore')
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================================
# 策略参数设置区
TS_KEY = pw.TUSHARE_TOKEN
# ============================================================================

class StrategyConfig:
    """策略参数配置类"""

    # 获取今日和昨天日期
    TODAY = datetime.now()
    YESTERDAY = TODAY - timedelta(days=1)

    # 基础参数
    START_DATE = '20240101'  # 开始日期
    END_DATE = YESTERDAY.strftime('%Y%m%d')  # 结束日期设为昨天（确保数据完整）
    INDEX_CODE = '000001.SH'  # 基准指数代码（上证指数）
    INITIAL_CAPITAL = 1000000  # 初始资金
    TRANSACTION_COST = 0.001  # 交易成本（单边）

    # 推波助澜模型参数
    THRESHOLD_MU = 9.5  # 近似涨跌停阈值(%)，研报使用9.5%
    SHORT_LEN = 30  # 短期均线长度，研报使用30日
    LONG_LEN = 100  # 长期均线长度，研报使用100日
    SIGNAL_THRESHOLD = 1.15  # 信号阈值，研报使用1.15

    # 数据参数
    CACHE_DIR = './data_parquet/'  # Parquet数据缓存目录
    STOCK_LIST_DIR = './stock_lists/'  # 股票列表输出目录
    FORCE_UPDATE = False  # 是否强制更新数据
    USE_CACHE = True  # 是否使用缓存数据

    # 回测参数
    SAMPLE_SPLIT_DATE = '20230601'  # 样本内外分割日期
    RISK_FREE_RATE = 0.02  # 无风险利率

    # 输出设置
    OUTPUT_DIR = './output/'  # 输出目录


# ============================================================================
# 数据管理类 - Parquet版本
# ============================================================================

class DataManager:
    """数据管理类，负责数据下载、缓存和读取（Parquet格式）"""

    def __init__(self, pro_api, cache_dir=StrategyConfig.CACHE_DIR):
        self.pro = pro_api
        self.cache_dir = cache_dir
        self.stock_list_dir = StrategyConfig.STOCK_LIST_DIR
        self.ensure_dirs()

    def ensure_dirs(self):
        """确保缓存目录存在"""
        for dir_path in [self.cache_dir, self.stock_list_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def get_cache_path(self, data_type, **kwargs):
        """获取Parquet缓存文件路径"""
        if data_type == 'stock_basic':
            return os.path.join(self.cache_dir, 'stock_basic.parquet')
        elif data_type == 'daily_data':
            date = kwargs.get('date')
            return os.path.join(self.cache_dir, 'daily', f'daily_{date}.parquet')
        elif data_type == 'index_data':
            index_code = kwargs.get('index_code')
            start_date = kwargs.get('start_date', '')
            end_date = kwargs.get('end_date', '')
            return os.path.join(self.cache_dir, 'index', f'index_{index_code}_{start_date}_{end_date}.parquet')
        elif data_type == 'push_wave':
            start = kwargs.get('start_date', '')
            end = kwargs.get('end_date', '')
            mu = kwargs.get('mu', '9.5')
            return os.path.join(self.cache_dir, 'push_wave', f'push_wave_{start}_{end}_mu{mu}.parquet')
        elif data_type == 'limit_stocks':
            date = kwargs.get('date', '')
            mu = kwargs.get('mu', '9.5')
            return os.path.join(self.stock_list_dir, f'limit_stocks_{date}_mu{mu}.parquet')
        else:
            raise ValueError(f"未知的数据类型: {data_type}")

    def should_update_data(self, cache_path, max_age_days=1):
        """判断是否需要更新数据（股票数据每日更新）"""
        if not StrategyConfig.USE_CACHE or StrategyConfig.FORCE_UPDATE:
            return True

        if not os.path.exists(cache_path):
            return True

        # 检查文件是否过期（股票数据每日更新）
        if max_age_days:
            file_mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
            if (datetime.now() - file_mtime).days > max_age_days:
                return True

        return False

    def get_stock_basic(self, force_update=False):
        """获取股票基本信息，保存为Parquet格式"""
        cache_path = self.get_cache_path('stock_basic')

        if not force_update and os.path.exists(cache_path) and StrategyConfig.USE_CACHE:
            print(f"从Parquet缓存读取股票基本信息: {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"读取Parquet缓存失败: {e}")

        print("下载股票基本信息...")
        try:
            stock_basic = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name,list_date,delist_date,industry,market'
            )

            if not stock_basic.empty:
                # 确保目录存在
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

                # 保存为Parquet格式
                stock_basic.to_parquet(
                    cache_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )

            print(f"股票基本信息下载完成，共{len(stock_basic)}只股票，已保存为Parquet格式")
            return stock_basic
        except Exception as e:
            print(f"获取股票基本信息失败: {e}")
            return pd.DataFrame()

    def get_daily_data(self, trade_date, force_update=False):
        """获取指定日期的日行情数据（Parquet格式）"""
        if isinstance(trade_date, datetime):
            date_str = trade_date.strftime('%Y%m%d')
        elif isinstance(trade_date, pd.Timestamp):
            date_str = trade_date.strftime('%Y%m%d')
        else:
            date_str = str(trade_date)

        cache_path = self.get_cache_path('daily_data', date=date_str)

        if not force_update and os.path.exists(cache_path) and StrategyConfig.USE_CACHE:
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"读取Parquet缓存失败 {cache_path}: {e}")
                return pd.DataFrame()

        print(f"下载{date_str}日行情数据...")
        try:
            daily_data = self.pro.daily(trade_date=date_str)

            if not daily_data.empty:
                # 优化数据类型
                for col in ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']:
                    if col in daily_data.columns:
                        daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')

                # 确保目录存在
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)

                # 保存为Parquet格式
                daily_data.to_parquet(
                    cache_path,
                    engine='pyarrow',
                    compression='snappy',
                    index=False
                )
                print(f"  {date_str}数据下载完成，{len(daily_data)}条记录，已保存为Parquet格式")
            else:
                print(f"  {date_str}无数据")

            return daily_data
        except Exception as e:
            print(f"获取{date_str}数据失败: {e}")
            return pd.DataFrame()

    def batch_download_daily_data(self, start_date, end_date, update_existing=False):
        """批量下载日行情数据（保存为Parquet格式）"""
        dates = pd.date_range(start_date, end_date, freq='B')
        total_dates = len(dates)

        print(f"开始批量下载日行情数据，共{total_dates}个交易日")
        print(f"数据将保存为Parquet格式到: {self.cache_dir}")

        # 创建存储目录
        daily_dir = os.path.join(self.cache_dir, 'daily')
        os.makedirs(daily_dir, exist_ok=True)

        downloaded = 0
        skipped = 0
        failed = 0

        for i, date in enumerate(dates):
            if i % 20 == 0 or i == total_dates - 1:
                print(f"进度: {i + 1}/{total_dates} ({date.strftime('%Y-%m-%d')})")

            date_str = date.strftime('%Y%m%d')
            cache_path = os.path.join(daily_dir, f'daily_{date_str}.parquet')

            # 如果文件存在且不强制更新，跳过
            if os.path.exists(cache_path) and not update_existing:
                skipped += 1
                continue

            try:
                daily_data = self.pro.daily(trade_date=date_str)

                if not daily_data.empty:
                    # 优化数据类型
                    for col in ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']:
                        if col in daily_data.columns:
                            daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')

                    # 保存为Parquet
                    daily_data.to_parquet(
                        cache_path,
                        engine='pyarrow',
                        compression='snappy',
                        index=False
                    )
                    downloaded += 1

                # 避免请求过于频繁
                if downloaded % 50 == 0 and downloaded > 0:
                    import time
                    time.sleep(2)

            except Exception as e:
                print(f"下载{date_str}数据失败: {e}")
                failed += 1

        print(f"批量下载完成！下载{downloaded}个，跳过{skipped}个，失败{failed}个")
        print(f"所有数据已保存为Parquet格式在: {daily_dir}")

    def load_cached_daily_data_for_date_range(self, start_date, end_date):
        """加载指定日期范围内的所有缓存日行情数据"""
        dates = pd.date_range(start_date, end_date, freq='B')
        all_data = []

        daily_dir = os.path.join(self.cache_dir, 'daily')
        if not os.path.exists(daily_dir):
            print(f"目录不存在: {daily_dir}")
            return pd.DataFrame()

        print(f"从Parquet缓存加载日行情数据，时间范围: {start_date} 到 {end_date}")

        for i, date in enumerate(dates):
            if i % 50 == 0 and i > 0:
                print(f"加载进度: {i}/{len(dates)}")

            date_str = date.strftime('%Y%m%d')
            cache_path = os.path.join(daily_dir, f'daily_{date_str}.parquet')

            if os.path.exists(cache_path):
                try:
                    daily_data = pd.read_parquet(cache_path)
                    if not daily_data.empty:
                        # 添加交易日期列
                        daily_data['trade_date'] = pd.to_datetime(date_str)
                        all_data.append(daily_data)
                except Exception as e:
                    print(f"读取{date_str}数据失败: {e}")

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            print(f"数据加载完成，共{len(combined_data)}条记录")
            return combined_data
        else:
            print("没有找到缓存数据")
            return pd.DataFrame()

    def save_limit_stocks(self, date, limit_up_codes, limit_down_codes, threshold_mu):
        """
        保存涨停/跌停股票列表到Parquet文件

        参数:
        ----------
        date : datetime or str
            交易日
        limit_up_codes : list
            涨停股票代码列表
        limit_down_codes : list
            跌停股票代码列表
        threshold_mu : float
            涨跌停阈值(%)
        """
        if isinstance(date, (datetime, pd.Timestamp)):
            date_str = date.strftime('%Y%m%d')
        else:
            date_str = str(date)

        cache_path = self.get_cache_path('limit_stocks', date=date_str, mu=str(threshold_mu))

        # 创建涨停股票DataFrame
        if limit_up_codes:
            limit_up_df = pd.DataFrame({
                'ts_code': limit_up_codes,
                'limit_type': 'up',
                'trade_date': pd.to_datetime(date_str),
                'threshold_mu': threshold_mu
            })
        else:
            limit_up_df = pd.DataFrame(columns=['ts_code', 'limit_type', 'trade_date', 'threshold_mu'])

        # 创建跌停股票DataFrame
        if limit_down_codes:
            limit_down_df = pd.DataFrame({
                'ts_code': limit_down_codes,
                'limit_type': 'down',
                'trade_date': pd.to_datetime(date_str),
                'threshold_mu': threshold_mu
            })
        else:
            limit_down_df = pd.DataFrame(columns=['ts_code', 'limit_type', 'trade_date', 'threshold_mu'])

        # 合并两个DataFrame
        if not limit_up_df.empty or not limit_down_df.empty:
            limit_stocks_df = pd.concat([limit_up_df, limit_down_df], ignore_index=True)

            # 保存为Parquet格式
            limit_stocks_df.to_parquet(
                cache_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

            # 打印保存信息
            print(f"  涨停股票: {len(limit_up_codes)}只, 跌停股票: {len(limit_down_codes)}只")

            return cache_path
        else:
            print(f"  {date_str}: 无涨停跌停股票")
            return None

    def load_limit_stocks_for_date_range(self, start_date, end_date, threshold_mu):
        """
        加载指定日期范围内的涨停跌停股票数据

        参数:
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        threshold_mu : float
            涨跌停阈值(%)

        返回:
        ----------
        pd.DataFrame
            涨停跌停股票数据
        """
        dates = pd.date_range(start_date, end_date, freq='B')
        all_limit_stocks = []

        for date in dates:
            date_str = date.strftime('%Y%m%d')
            cache_path = self.get_cache_path('limit_stocks', date=date_str, mu=str(threshold_mu))

            if os.path.exists(cache_path):
                try:
                    limit_data = pd.read_parquet(cache_path)
                    if not limit_data.empty:
                        all_limit_stocks.append(limit_data)
                except Exception as e:
                    print(f"读取{date_str}涨停跌停数据失败: {e}")

        if all_limit_stocks:
            combined_data = pd.concat(all_limit_stocks, ignore_index=True)
            print(f"加载涨停跌停股票数据完成，共{len(combined_data)}条记录")
            return combined_data
        else:
            print("没有找到涨停跌停股票数据")
            return pd.DataFrame()

    def get_index_data(self, index_code, start_date, end_date, force_update=False):
        """获取指数数据（Parquet格式）"""
        cache_path = self.get_cache_path('index_data',
                                         index_code=index_code,
                                         start_date=start_date,
                                         end_date=end_date)

        # 检查缓存是否需要更新
        if not force_update and os.path.exists(cache_path) and StrategyConfig.USE_CACHE:
            try:
                print(f"从Parquet缓存读取指数数据: {cache_path}")
                index_data = pd.read_parquet(cache_path)

                # 设置索引
                if 'trade_date' in index_data.columns:
                    index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
                    index_data.set_index('trade_date', inplace=True)
                    index_data.sort_index(inplace=True)
                    index_data['return'] = index_data['close'].pct_change()

                return index_data
            except Exception as e:
                print(f"读取Parquet缓存失败: {e}")

        print(f"下载指数{index_code}数据，时间范围: {start_date} 到 {end_date}")
        try:
            # 对于不同的指数代码，可能需要不同的API
            if index_code.endswith('.SH') or index_code.endswith('.SZ'):
                # 使用指数日线数据接口
                index_data = self.pro.index_daily(
                    ts_code=index_code,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                # 对于万得全A等指数，可能需要使用其他接口
                # 这里先尝试通用接口
                index_data = self.pro.index_daily(
                    ts_code=index_code,
                    start_date=start_date,
                    end_date=end_date
                )

            if index_data.empty:
                print(f"警告: 无法获取指数{index_code}的数据")
                return pd.DataFrame()

            # 处理数据
            index_data['trade_date'] = pd.to_datetime(index_data['trade_date'])
            index_data.set_index('trade_date', inplace=True)
            index_data.sort_index(inplace=True)
            index_data['return'] = index_data['close'].pct_change()

            # 确保目录存在并保存为Parquet
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            # 重置索引以便保存
            index_data_reset = index_data.reset_index()
            index_data_reset.to_parquet(
                cache_path,
                engine='pyarrow',
                compression='snappy',
                index=False
            )

            print(f"指数数据下载完成，共{len(index_data)}个交易日，已保存为Parquet格式")

            return index_data
        except Exception as e:
            print(f"获取指数数据失败: {e}")
            return pd.DataFrame()


# ============================================================================
# 推波助澜择时模型（Hull移动平均线版本）
# ============================================================================

class PushWaveModel:
    """
    推波助澜择时模型（Hull移动平均线版本）
    """

    def __init__(self,
                 data_manager: DataManager,
                 threshold: float = StrategyConfig.THRESHOLD_MU,
                 short_len: int = StrategyConfig.SHORT_LEN,
                 long_len: int = StrategyConfig.LONG_LEN,
                 signal_threshold: float = StrategyConfig.SIGNAL_THRESHOLD):
        """
        初始化模型参数
        """
        self.data_manager = data_manager
        self.threshold = threshold / 100  # 转换为小数
        self.short_len = short_len
        self.long_len = long_len
        self.signal_threshold = signal_threshold
        self.stock_basic = None
        self.trade_records = []  # 用于记录交易历史
        self.output_dir = StrategyConfig.OUTPUT_DIR
        self.last_metrics = None  # 存储最后的绩效指标
        self.daily_limit_stocks = {}  # 存储每日涨停跌停股票数据
        self.ensure_output_dir()

    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def calculate_wma(self, df: pd.DataFrame, index: int, n: int, col: str) -> float:
        """
        计算加权移动平均（WMA）

        参数:
        ----------
        df : pd.DataFrame
            数据框
        index : int
            当前索引位置
        n : int
            窗口大小
        col : str
            列名

        返回:
        ----------
        float
            加权移动平均值
        """
        wma = 0.0

        # 如果索引小于n，调整窗口大小
        if index < n:
            n = index + 1

        # 计算加权移动平均
        for i in range(n):
            w = (i + 1) / (0.5 * n * (n + 1))
            wma += w * df.iloc[index - n + 1 + i][col]

        return wma

    def calculate_hma(self, series: pd.Series, n: int) -> pd.Series:
        """
        计算Hull移动平均线（HMA）

        参数:
        ----------
        series : pd.Series
            原始序列
        n : int
            窗口大小

        返回:
        ----------
        pd.Series
            HMA序列
        """
        # 将序列转换为DataFrame以便计算
        df = pd.DataFrame({'value': series.values}, index=series.index)
        n = int(n)

        # 计算HMA_raw
        df['hma_raw'] = 0.0

        for i in range(len(df)):
            # 计算HMA_raw = 2 * WMA(n/2) - WMA(n)
            wma_half = self.calculate_wma(df, i, int(n / 2), 'value')
            wma_full = self.calculate_wma(df, i, n, 'value')
            df.iloc[i, df.columns.get_loc('hma_raw')] = 2 * wma_half - wma_full

        # 计算最终的HMA = WMA(hma_raw, sqrt(n))
        df['hma'] = 0.0
        sqrt_n = int(math.sqrt(n))

        for i in range(len(df)):
            if i >= sqrt_n - 1:  # 确保有足够的数据计算WMA
                df.iloc[i, df.columns.get_loc('hma')] = self.calculate_wma(
                    df, i, sqrt_n, 'hma_raw'
                )

        return df['hma']

    def get_stock_basic_info(self):
        """获取股票基本信息，排除ST股票"""
        if self.stock_basic is None:
            self.stock_basic = self.data_manager.get_stock_basic()

        if self.stock_basic.empty:
            return self.stock_basic

        # 排除ST股票（名称中包含ST）
        non_st_mask = ~self.stock_basic['name'].str.contains('ST|退市', na=False)
        return self.stock_basic[non_st_mask].copy()

    def calculate_push_wave_ratio(self, start_date: str, end_date: str, use_cache: bool = True,
                                  save_limit_stocks: bool = True) -> pd.DataFrame:
        """
        计算推波助澜比率（涨跌停剪刀差）
        """
        # 检查缓存
        cache_path = self.data_manager.get_cache_path('push_wave',
                                                      start_date=start_date,
                                                      end_date=end_date,
                                                      mu=str(self.threshold * 100))

        if use_cache and os.path.exists(cache_path) and StrategyConfig.USE_CACHE:
            print(f"从Parquet缓存读取推波助澜比率数据: {cache_path}")
            try:
                results_df = pd.read_parquet(cache_path)
                if 'trade_date' in results_df.columns:
                    results_df['trade_date'] = pd.to_datetime(results_df['trade_date'])
                    results_df.set_index('trade_date', inplace=True)
                return results_df
            except Exception as e:
                print(f"读取Parquet缓存失败: {e}")

        print(f"计算推波助澜比率，时间范围：{start_date} 到 {end_date}")
        print(f"涨跌停阈值: μ={self.threshold * 100}%")

        # 从本地Parquet文件加载所有数据
        all_daily_data = self.data_manager.load_cached_daily_data_for_date_range(start_date, end_date)

        if all_daily_data.empty:
            raise ValueError("无法加载日行情数据，请检查数据是否已下载")

        # 获取股票基本信息（排除ST）
        stock_basic = self.get_stock_basic_info()
        if stock_basic.empty:
            raise ValueError("股票基本信息为空，请检查数据获取")

        valid_stocks = set(stock_basic['ts_code'].tolist())
        print(f"有效股票数量（非ST）: {len(valid_stocks)}")

        # 过滤有效股票
        all_daily_data = all_daily_data[all_daily_data['ts_code'].isin(valid_stocks)].copy()
        all_daily_data = all_daily_data[all_daily_data['pct_chg'].notna()].copy()

        # 按日期分组计算
        results = []

        # 获取所有唯一日期并按日期排序
        unique_dates = pd.to_datetime(all_daily_data['trade_date'].unique())
        unique_dates = sorted(unique_dates)

        for i, date in enumerate(unique_dates):
            if i % 50 == 0 or i == len(unique_dates) - 1:
                print(f"进度：{i + 1}/{len(unique_dates)} ({date.strftime('%Y-%m-%d')})")

            # 使用日期字符串进行比较
            date_str = date.strftime('%Y-%m-%d')
            date_data = all_daily_data[all_daily_data['trade_date'].dt.strftime('%Y-%m-%d') == date_str].copy()

            if len(date_data) == 0:
                continue

            # 涨停股票数量（涨幅大于阈值）
            limit_up_mask = date_data['pct_chg'] > self.threshold * 100
            limit_up_data = date_data[limit_up_mask]
            limit_up_count = len(limit_up_data)
            limit_up_codes = limit_up_data['ts_code'].tolist()

            # 跌停股票数量（跌幅小于负阈值）
            limit_down_mask = date_data['pct_chg'] < -self.threshold * 100
            limit_down_data = date_data[limit_down_mask]
            limit_down_count = len(limit_down_data)
            limit_down_codes = limit_down_data['ts_code'].tolist()

            # 总股票数量
            total_count = len(date_data)

            if total_count > 0:
                # 涨停比率
                limit_up_ratio = limit_up_count / total_count

                # 跌停比率
                limit_down_ratio = limit_down_count / total_count

                # 推波助澜比率（剪刀差）
                push_wave_ratio = limit_up_ratio - limit_down_ratio

                results.append({
                    'trade_date': date,
                    'limit_up_ratio': limit_up_ratio,
                    'limit_down_ratio': limit_down_ratio,
                    'push_wave_ratio': push_wave_ratio,
                    'total_stocks': total_count,
                    'limit_up_count': limit_up_count,
                    'limit_down_count': limit_down_count
                })

                # 保存涨停跌停股票列表
                if save_limit_stocks:
                    # 保存到Parquet文件
                    saved_path = self.data_manager.save_limit_stocks(
                        date=date,
                        limit_up_codes=limit_up_codes,
                        limit_down_codes=limit_down_codes,
                        threshold_mu=self.threshold * 100
                    )

                    # 保存到内存供后续使用
                    self.daily_limit_stocks[date] = {
                        'limit_up_codes': limit_up_codes,
                        'limit_down_codes': limit_down_codes,
                        'limit_up_count': limit_up_count,
                        'limit_down_count': limit_down_count
                    }

        if not results:
            raise ValueError("无法计算推波助澜比率，请检查数据获取")

        results_df = pd.DataFrame(results)
        results_df.set_index('trade_date', inplace=True)
        results_df.sort_index(inplace=True)

        # 确保目录存在并缓存结果为Parquet格式
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        results_df.reset_index().to_parquet(
            cache_path,
            engine='pyarrow',
            compression='snappy',
            index=False
        )

        print(f"推波助澜比率计算完成，共{len(results_df)}个交易日")
        print(f"涨停跌停股票列表已保存到: {self.data_manager.stock_list_dir}")

        return results_df

    def load_daily_limit_stocks(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        加载指定日期范围内的涨停跌停股票数据
        """
        return self.data_manager.load_limit_stocks_for_date_range(
            start_date=start_date,
            end_date=end_date,
            threshold_mu=self.threshold * 100
        )

    def calculate_moving_averages(self, push_wave_series: pd.Series) -> pd.DataFrame:
        """
        计算推波助澜比率的Hull移动平均线
        """
        df = pd.DataFrame(push_wave_series)
        df.columns = ['push_wave_ratio_raw']

        # 计算HMA短期均线（HMA30）
        df['hma_short'] = self.calculate_hma(
            series=df['push_wave_ratio_raw'],
            n=self.short_len
        )

        # 计算HMA长期均线（HMA100）
        df['hma_long'] = self.calculate_hma(
            series=df['push_wave_ratio_raw'],
            n=self.long_len
        )

        # 计算HMA比率
        df['hma_ratio'] = df['hma_short'] / df['hma_long']
        df['hma_ratio'] = df['hma_ratio'].replace([np.inf, -np.inf], np.nan)

        # 添加HMA差值和百分比差值
        df['hma_diff'] = df['hma_short'] - df['hma_long']
        df['hma_pct_diff'] = (df['hma_short'] - df['hma_long']) / df['hma_long'].abs().replace(0, np.nan)

        return df

    def generate_trading_signals(self, ma_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        """
        df = pd.DataFrame(index=ma_data.index)

        # 初始化
        df['position'] = 0
        df['signal_type'] = 'hold'
        df['signal_reason'] = ''

        # 定义买入条件
        buy_condition = (
                (ma_data['hma_ratio'] > self.signal_threshold) &
                (ma_data['hma_short'] > 0) &
                (ma_data['hma_long'] > 0)
        )

        # 定义卖出条件
        sell_condition = (
                (ma_data['hma_ratio'] <= self.signal_threshold) |
                (ma_data['hma_short'] <= 0) |
                (ma_data['hma_long'] <= 0)
        )

        # 确保条件有效
        buy_condition = buy_condition.fillna(False)
        sell_condition = sell_condition.fillna(False)

        # 计算信号
        for i in range(1, len(df)):
            prev_position = df['position'].iloc[i - 1]

            if prev_position == 0:  # 当前空仓
                if buy_condition.iloc[i]:
                    df.at[df.index[i], 'position'] = 1
                    df.at[df.index[i], 'signal_type'] = 'buy'

                    # 记录买入原因
                    reasons = []
                    if ma_data['hma_ratio'].iloc[i] > self.signal_threshold:
                        reasons.append(f"HMA比率>{self.signal_threshold}")
                    if ma_data['hma_short'].iloc[i] > 0:
                        reasons.append("HMA短期>0")
                    if ma_data['hma_long'].iloc[i] > 0:
                        reasons.append("HMA长期>0")
                    df.at[df.index[i], 'signal_reason'] = ', '.join(reasons)
                else:
                    df.at[df.index[i], 'position'] = 0
                    df.at[df.index[i], 'signal_type'] = 'hold'

            else:  # 当前持有多仓
                if sell_condition.iloc[i]:
                    df.at[df.index[i], 'position'] = 0
                    df.at[df.index[i], 'signal_type'] = 'sell'

                    # 记录卖出原因
                    reasons = []
                    if ma_data['hma_ratio'].iloc[i] <= self.signal_threshold:
                        reasons.append(f"HMA比率≤{self.signal_threshold}")
                    if ma_data['hma_short'].iloc[i] <= 0:
                        reasons.append("HMA短期≤0")
                    if ma_data['hma_long'].iloc[i] <= 0:
                        reasons.append("HMA长期≤0")
                    df.at[df.index[i], 'signal_reason'] = ', '.join(reasons)
                else:
                    df.at[df.index[i], 'position'] = 1
                    df.at[df.index[i], 'signal_type'] = 'hold'

        return df

    def backtest_with_trading(self,
                              signals_df: pd.DataFrame,
                              index_returns: pd.Series,
                              initial_capital: float = StrategyConfig.INITIAL_CAPITAL,
                              transaction_cost: float = StrategyConfig.TRANSACTION_COST) -> pd.DataFrame:
        """
        回测函数 - 包含详细的交易记录
        """
        # 对齐数据
        aligned_data = pd.DataFrame({
            'position': signals_df['position'],
            'signal_type': signals_df['signal_type'],
            'signal_reason': signals_df['signal_reason'],
            'return': index_returns
        }).dropna()

        if aligned_data.empty:
            raise ValueError("回测数据为空")

        # 初始化
        aligned_data['position_lagged'] = aligned_data['position'].shift(1).fillna(0)
        aligned_data['strategy_return'] = 0.0
        aligned_data['transaction_cost'] = 0.0
        aligned_data['trade_count'] = 0

        # 记录交易
        trade_count = 0
        trade_details = []

        for i in range(1, len(aligned_data)):
            # 计算收益（使用前一天的仓位）
            aligned_data.loc[aligned_data.index[i], 'strategy_return'] = (
                    aligned_data['position_lagged'].iloc[i] * aligned_data['return'].iloc[i]
            )

            # 检查是否有交易发生
            if aligned_data['signal_type'].iloc[i] in ['buy', 'sell']:
                trade_count += 1

                # 扣除交易成本
                cost = transaction_cost * initial_capital
                aligned_data.loc[aligned_data.index[i], 'transaction_cost'] = cost
                aligned_data.loc[aligned_data.index[i], 'strategy_return'] -= transaction_cost

                # 记录交易详情
                trade_details.append({
                    'date': aligned_data.index[i],
                    'type': aligned_data['signal_type'].iloc[i],
                    'reason': aligned_data['signal_reason'].iloc[i],
                    'position_before': aligned_data['position'].iloc[i - 1],
                    'position_after': aligned_data['position'].iloc[i],
                    'cost': cost
                })

            aligned_data.loc[aligned_data.index[i], 'trade_count'] = trade_count

        # 计算净值
        aligned_data['benchmark_nav'] = (1 + aligned_data['return']).cumprod() * initial_capital

        # 考虑交易成本后的策略净值
        net_return = aligned_data['strategy_return'] - aligned_data['transaction_cost'] / initial_capital
        aligned_data['strategy_nav'] = (1 + net_return).cumprod() * initial_capital

        # 计算每日收益率
        aligned_data['strategy_daily_return'] = aligned_data['strategy_return']

        # 计算累计收益率
        aligned_data['strategy_cum_return'] = (1 + aligned_data['strategy_daily_return']).cumprod() - 1
        aligned_data['benchmark_cum_return'] = (1 + aligned_data['return']).cumprod() - 1

        # 保存交易记录
        self.trade_records = trade_details

        return aligned_data

    def calculate_performance_metrics(self, backtest_data: pd.DataFrame) -> Dict:
        """
        计算绩效指标
        """
        if len(backtest_data) < 2:
            return {}

        # 总收益率
        total_return = backtest_data['strategy_nav'].iloc[-1] / backtest_data['strategy_nav'].iloc[0] - 1
        benchmark_return = backtest_data['benchmark_nav'].iloc[-1] / backtest_data['benchmark_nav'].iloc[0] - 1

        # 年化收益率
        years = len(backtest_data) / 252
        if years > 0:
            annual_return = (1 + total_return) ** (1 / years) - 1
            benchmark_annual = (1 + benchmark_return) ** (1 / years) - 1
        else:
            annual_return = total_return
            benchmark_annual = benchmark_return

        # 年化波动率
        daily_returns = backtest_data['strategy_daily_return'].dropna()
        if len(daily_returns) > 0:
            annual_volatility = daily_returns.std() * np.sqrt(252)
        else:
            annual_volatility = 0

        # 夏普比率
        risk_free_rate = StrategyConfig.RISK_FREE_RATE
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0

        # 最大回撤
        strategy_nav = backtest_data['strategy_nav']
        rolling_max = strategy_nav.expanding().max()
        drawdown = (strategy_nav - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 最大回撤开始和结束日期
        if max_drawdown < 0:
            drawdown_end_idx = drawdown.idxmin()
            drawdown_start_idx = strategy_nav[:drawdown_end_idx].idxmax()
            max_drawdown_start = drawdown_start_idx.strftime('%Y-%m-%d')
            max_drawdown_end = drawdown_end_idx.strftime('%Y-%m-%d')
        else:
            max_drawdown_start = 'N/A'
            max_drawdown_end = 'N/A'

        # 胜率（多头交易胜率）
        long_returns = backtest_data[backtest_data['position_lagged'] == 1]['strategy_daily_return'].dropna()
        if len(long_returns) > 0:
            win_rate = len(long_returns[long_returns > 0]) / len(long_returns)
        else:
            win_rate = 0

        # 盈亏比
        winning_returns = long_returns[long_returns > 0].mean() if len(long_returns[long_returns > 0]) > 0 else 0
        losing_returns = abs(long_returns[long_returns < 0].mean()) if len(long_returns[long_returns < 0]) > 0 else 0
        profit_loss_ratio = winning_returns / losing_returns if losing_returns > 0 else 0

        # 交易次数
        total_trades = len(self.trade_records)
        annual_trades = total_trades / years if years > 0 else 0

        # 多头持有期
        long_periods = (backtest_data['position'] == 1)
        if long_periods.any():
            long_changes = (long_periods != long_periods.shift(1)).cumsum()
            long_durations = backtest_data.groupby(long_changes)['position'].apply(
                lambda x: len(x) if x.iloc[0] == 1 else 0
            )
            long_durations = long_durations[long_durations > 0]
            avg_long_duration = long_durations.mean() if len(long_durations) > 0 else 0
        else:
            avg_long_duration = 0

        # 空头持有期
        short_periods = (backtest_data['position'] == 0)
        if short_periods.any():
            short_changes = (short_periods != short_periods.shift(1)).cumsum()
            short_durations = backtest_data.groupby(short_changes)['position'].apply(
                lambda x: len(x) if x.iloc[0] == 0 else 0
            )
            short_durations = short_durations[short_durations > 0]
            avg_short_duration = short_durations.mean() if len(short_durations) > 0 else 0
        else:
            avg_short_duration = 0

        # 卡玛比率
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0

        # 月度胜率
        try:
            monthly_returns = backtest_data['strategy_daily_return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1
            )
            monthly_win_rate = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
        except:
            monthly_win_rate = 0

        # 周度胜率
        try:
            weekly_returns = backtest_data['strategy_daily_return'].resample('W-FRI').apply(
                lambda x: (1 + x).prod() - 1
            )
            weekly_win_rate = (weekly_returns > 0).mean() if len(weekly_returns) > 0 else 0
        except:
            weekly_win_rate = 0

        metrics = {
            '年化收益率': annual_return,
            '基准年化收益率': benchmark_annual,
            '夏普比率': sharpe_ratio,
            '最大回撤': max_drawdown,
            '最大回撤开始': max_drawdown_start,
            '最大回撤结束': max_drawdown_end,
            '胜率': win_rate,
            '盈亏比': profit_loss_ratio,
            '年化波动率': annual_volatility,
            '总交易次数': total_trades,
            '年均交易次数': annual_trades,
            '平均多头持有期(天)': avg_long_duration,
            '平均空头持有期(天)': avg_short_duration,
            '卡玛比率': calmar_ratio,
            '总收益率': total_return,
            '基准总收益率': benchmark_return,
            '月度胜率': monthly_win_rate,
            '周度胜率': weekly_win_rate,
            '总交易日数': len(backtest_data)
        }

        # 保存指标供导出使用
        self.last_metrics = metrics

        return metrics

    def print_trading_summary(self):
        """打印交易摘要"""
        if not self.trade_records:
            print("没有交易记录")
            return

        print("\n" + "=" * 60)
        print("交易记录摘要")
        print("=" * 60)

        total_trades = len(self.trade_records)
        buy_trades = len([t for t in self.trade_records if t['type'] == 'buy'])
        sell_trades = len([t for t in self.trade_records if t['type'] == 'sell'])

        print(f"总交易次数: {total_trades}次")
        print(f"买入次数: {buy_trades}次")
        print(f"卖出次数: {sell_trades}次")

        # 按原因统计卖出交易
        sell_reasons = {}
        for trade in self.trade_records:
            if trade['type'] == 'sell':
                reason = trade['reason']
                sell_reasons[reason] = sell_reasons.get(reason, 0) + 1

        if sell_reasons:
            print("\n卖出原因统计:")
            for reason, count in sell_reasons.items():
                print(f"  {reason}: {count}次")

        # 显示前10次交易详情
        print(f"\n前10次交易详情:")
        print("-" * 80)
        print(f"{'日期':<12} {'类型':<6} {'原因':<30} {'前仓位':<8} {'后仓位':<8}")
        print("-" * 80)

        for i, trade in enumerate(self.trade_records[:10]):
            print(f"{trade['date'].strftime('%Y-%m-%d'):<12} "
                  f"{trade['type']:<6} "
                  f"{trade['reason'][:28]:<30} "
                  f"{trade['position_before']:<8} "
                  f"{trade['position_after']:<8}")

    def plot_results(self, backtest_data: pd.DataFrame, ma_data: pd.DataFrame,
                     push_wave_data: pd.DataFrame, signals_df: pd.DataFrame):
        """
        绘制结果图表
        """
        try:
            fig, axes = plt.subplots(5, 1, figsize=(16, 20))

            # 1. 净值曲线
            ax1 = axes[0]
            ax1.plot(backtest_data.index, backtest_data['strategy_nav'],
                     label='Strategy NAV', color='red', linewidth=2)
            ax1.plot(backtest_data.index, backtest_data['benchmark_nav'],
                     label='Benchmark NAV', color='blue', alpha=0.7)
            ax1.set_title(f'Push Wave Timing Strategy NAV Curve (mu={self.threshold * 100}%)', fontsize=14)
            ax1.set_ylabel('NAV')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. 累计收益率对比
            ax2 = axes[1]
            strategy_cum_return = backtest_data['strategy_nav'] / backtest_data['strategy_nav'].iloc[0] - 1
            benchmark_cum_return = backtest_data['benchmark_nav'] / backtest_data['benchmark_nav'].iloc[0] - 1

            ax2.plot(backtest_data.index, strategy_cum_return * 100,
                     label='Strategy Cumulative Return', color='red', linewidth=2)
            ax2.plot(backtest_data.index, benchmark_cum_return * 100,
                     label='Benchmark Cumulative Return', color='blue', alpha=0.7)
            ax2.set_title('Cumulative Return Comparison', fontsize=14)
            ax2.set_ylabel('Cumulative Return (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 推波助澜比率及其HMA
            ax3 = axes[2]
            # 合并数据用于绘图
            aligned_data = ma_data.copy()
            aligned_data['push_wave_raw'] = push_wave_data['push_wave_ratio']

            ax3.plot(aligned_data.index, aligned_data['push_wave_raw'],
                     label='Push Wave Ratio', color='black', alpha=0.5, linewidth=0.5)
            ax3.plot(aligned_data.index, aligned_data['hma_short'],
                     label=f'HMA{self.short_len}', color='red', linewidth=1.5)
            ax3.plot(aligned_data.index, aligned_data['hma_long'],
                     label=f'HMA{self.long_len}', color='blue', linewidth=1.5)
            ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax3.set_title('Push Wave Ratio and Hull Moving Averages', fontsize=14)
            ax3.set_ylabel('Ratio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. HMA比率信号
            ax4 = axes[3]
            hma_ratio = ma_data['hma_ratio'].dropna()
            ax4.plot(hma_ratio.index, hma_ratio, label=f'HMA{self.short_len}/HMA{self.long_len}', color='green',
                     linewidth=1.5)
            ax4.axhline(y=self.signal_threshold, color='red', linestyle='--',
                        alpha=0.7, label=f'Threshold={self.signal_threshold}')
            ax4.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
            ax4.set_title('HMA Ratio and Signal Threshold', fontsize=14)
            ax4.set_ylabel('HMA Ratio')
            ax4.legend(loc='best')
            ax4.grid(True, alpha=0.3)

            # 5. 交易信号
            ax5 = axes[4]
            # 标记买入和卖出信号
            buy_signals = signals_df[signals_df['signal_type'] == 'buy']
            sell_signals = signals_df[signals_df['signal_type'] == 'sell']

            # 绘制仓位
            ax5.fill_between(backtest_data.index, 0, backtest_data['position'],
                             where=(backtest_data['position'] > 0),
                             color='green', alpha=0.3, label='Holding Position')

            # 标记买入点
            if not buy_signals.empty:
                ax5.scatter(buy_signals.index, [0.2] * len(buy_signals),
                            color='red', marker='^', s=100, label='Buy Signal', zorder=5)

            # 标记卖出点
            if not sell_signals.empty:
                ax5.scatter(sell_signals.index, [-0.2] * len(sell_signals),
                            color='blue', marker='v', s=100, label='Sell Signal', zorder=5)

            ax5.set_title('Trading Signals (Red Triangle: Buy, Blue Triangle: Sell, Green Area: Holding Position)', fontsize=14)
            ax5.set_xlabel('Date')
            ax5.set_ylabel('Signal')
            ax5.set_ylim(-0.5, 1.5)
            ax5.legend(loc='upper left')
            ax5.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"绘制图表时出错: {e}")

    def export_daily_positions_to_excel(self, backtest_data: pd.DataFrame,
                                        ma_data: pd.DataFrame = None,
                                        push_wave_data: pd.DataFrame = None,
                                        filename: str = None):
        """
        将每日持仓情况导出到Excel文件
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"push_wave_daily_positions.xlsx"

        filepath = os.path.join(self.output_dir, filename)

        print(f"\n导出每日持仓情况到Excel: {filepath}")

        # 加载涨停跌停股票数据
        limit_stocks_df = self.load_daily_limit_stocks(
            StrategyConfig.START_DATE,
            StrategyConfig.END_DATE
        )

        # 创建Excel写入器
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 1. 每日持仓和绩效数据
            position_data = backtest_data.copy()

            # 添加额外指标
            position_data['日期'] = position_data.index
            position_data['仓位'] = position_data['position']
            position_data['信号类型'] = position_data['signal_type']
            position_data['信号原因'] = position_data['signal_reason']
            position_data['策略日收益率'] = position_data['strategy_daily_return']
            position_data['基准日收益率'] = position_data['return']
            position_data['策略累计收益率'] = position_data['strategy_cum_return']
            position_data['基准累计收益率'] = position_data['benchmark_cum_return']
            position_data['策略净值'] = position_data['strategy_nav']
            position_data['基准净值'] = position_data['benchmark_nav']

            # 选择要导出的列
            export_cols = [
                '日期', '仓位', '信号类型', '信号原因',
                '策略日收益率', '基准日收益率',
                '策略累计收益率', '基准累计收益率',
                '策略净值', '基准净值'
            ]

            # 只保留存在的列
            existing_cols = [col for col in export_cols if col in position_data.columns]
            daily_positions = position_data[existing_cols].copy()

            # 格式化数值
            for col in daily_positions.columns:
                if '收益率' in col:
                    daily_positions[col] = daily_positions[col].apply(lambda x: f"{x:.4%}" if pd.notnull(x) else "")
                elif '净值' in col:
                    daily_positions[col] = daily_positions[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
                elif '仓位' in col:
                    daily_positions[col] = daily_positions[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "")

            # 保存到Excel
            daily_positions.to_excel(writer, sheet_name='每日持仓', index=False)
            print(f"  - 已导出每日持仓数据，共{len(daily_positions)}个交易日")

            # 2. 交易记录
            if self.trade_records:
                trade_df = pd.DataFrame(self.trade_records)

                # 格式化交易记录
                if not trade_df.empty:
                    trade_df['日期'] = trade_df['date'].dt.strftime('%Y-%m-%d')
                    trade_df['交易类型'] = trade_df['type']
                    trade_df['交易原因'] = trade_df['reason']
                    trade_df['交易前仓位'] = trade_df['position_before']
                    trade_df['交易后仓位'] = trade_df['position_after']
                    trade_df['交易成本'] = trade_df['cost'].apply(lambda x: f"{x:,.2f}")

                    trade_export_cols = ['日期', '交易类型', '交易原因',
                                         '交易前仓位', '交易后仓位', '交易成本']

                    trade_export_df = trade_df[trade_export_cols].copy()
                    trade_export_df.to_excel(writer, sheet_name='交易记录', index=False)
                    print(f"  - 已导出交易记录，共{len(trade_export_df)}笔交易")

            # 3. 技术指标数据
            if ma_data is not None and push_wave_data is not None:
                # 合并数据
                indicator_data = pd.DataFrame(index=push_wave_data.index)
                indicator_data['日期'] = indicator_data.index

                # 添加推波助澜相关指标
                if 'push_wave_ratio' in push_wave_data.columns:
                    indicator_data['推波助澜比率'] = push_wave_data['push_wave_ratio']

                if 'limit_up_ratio' in push_wave_data.columns:
                    indicator_data['涨停比率'] = push_wave_data['limit_up_ratio']

                if 'limit_down_ratio' in push_wave_data.columns:
                    indicator_data['跌停比率'] = push_wave_data['limit_down_ratio']

                # 添加HMA指标
                if 'hma_short' in ma_data.columns:
                    indicator_data['HMA短期'] = ma_data['hma_short']

                if 'hma_long' in ma_data.columns:
                    indicator_data['HMA长期'] = ma_data['hma_long']

                if 'hma_ratio' in ma_data.columns:
                    indicator_data['HMA比率'] = ma_data['hma_ratio']

                # 格式化数值
                for col in indicator_data.columns:
                    if col != '日期':
                        indicator_data[col] = indicator_data[col].apply(lambda x: f"{x:.6f}" if pd.notnull(x) else "")

                indicator_data.to_excel(writer, sheet_name='技术指标', index=False)
                print(f"  - 已导出技术指标数据")

            # 4. 涨停跌停股票数据
            if not limit_stocks_df.empty:
                # 4.1 涨停股票
                limit_up_df = limit_stocks_df[limit_stocks_df['limit_type'] == 'up'].copy()
                if not limit_up_df.empty:
                    limit_up_df['日期'] = pd.to_datetime(limit_up_df['trade_date'])
                    limit_up_df['股票代码'] = limit_up_df['ts_code']
                    limit_up_export = limit_up_df[['日期', '股票代码']].copy()

                    # 按日期分组，合并股票代码
                    limit_up_grouped = limit_up_export.groupby('日期')['股票代码'].apply(
                        lambda x: ', '.join(x)
                    ).reset_index()
                    limit_up_grouped['涨停数量'] = limit_up_grouped['股票代码'].apply(lambda x: len(x.split(', ')))

                    limit_up_grouped.to_excel(writer, sheet_name='涨停股票列表', index=False)
                    print(f"  - 已导出涨停股票列表，共{len(limit_up_grouped)}个交易日")

                # 4.2 跌停股票
                limit_down_df = limit_stocks_df[limit_stocks_df['limit_type'] == 'down'].copy()
                if not limit_down_df.empty:
                    limit_down_df['日期'] = pd.to_datetime(limit_down_df['trade_date'])
                    limit_down_df['股票代码'] = limit_down_df['ts_code']
                    limit_down_export = limit_down_df[['日期', '股票代码']].copy()

                    # 按日期分组，合并股票代码
                    limit_down_grouped = limit_down_export.groupby('日期')['股票代码'].apply(
                        lambda x: ', '.join(x)
                    ).reset_index()
                    limit_down_grouped['跌停数量'] = limit_down_grouped['股票代码'].apply(lambda x: len(x.split(', ')))

                    limit_down_grouped.to_excel(writer, sheet_name='跌停股票列表', index=False)
                    print(f"  - 已导出跌停股票列表，共{len(limit_down_grouped)}个交易日")

            # 5. 绩效汇总
            if self.last_metrics:
                metrics_df = pd.DataFrame(list(self.last_metrics.items()), columns=['指标', '值'])

                # 格式化绩效指标
                metrics_df['值'] = metrics_df['值'].apply(
                    lambda x: f"{x:.4%}" if isinstance(x, float) and '收益率' in metrics_df['指标'].iloc[0]
                    else f"{x:.4f}" if isinstance(x, float)
                    else str(x)
                )

                metrics_df.to_excel(writer, sheet_name='绩效汇总', index=False)
                print(f"  - 已导出绩效汇总")

        print(f"Excel文件已成功生成: {filepath}")
        return filepath

    def generate_html_report(self, backtest_data: pd.DataFrame,
                             push_wave_data: pd.DataFrame = None,
                             ma_data: pd.DataFrame = None,
                             filename: str = None) -> str:
        """
        生成HTML格式的策略报告

        参数:
        ----------
        backtest_data : pd.DataFrame
            回测数据
        push_wave_data : pd.DataFrame, optional
            推波助澜比率数据
        ma_data : pd.DataFrame, optional
            移动平均数据
        filename : str, optional
            输出文件名

        返回:
        ----------
        str
            HTML文件路径
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"strategy_dashboard.html"

        filepath = os.path.join(self.output_dir, filename)

        print(f"\n生成HTML策略报告: {filepath}")

        # 获取最新日期
        last_date = backtest_data.index[-1].strftime('%Y-%m-%d')

        # 获取今日策略回报
        today_return = backtest_data.iloc[-1]['strategy_daily_return']
        today_position = backtest_data.iloc[-1]['position']

        # 获取过去一个月的数据（约22个交易日）
        one_month_ago = backtest_data.index[-1] - pd.Timedelta(days=30)
        past_month_data = backtest_data[backtest_data.index >= one_month_ago]

        # 获取过去一年的数据（约252个交易日）
        one_year_ago = backtest_data.index[-1] - pd.Timedelta(days=365)
        past_year_data = backtest_data[backtest_data.index >= one_year_ago]

        # 计算过去一个月绩效指标
        past_month_metrics = self._calculate_period_metrics(past_month_data)

        # 计算过去一年绩效指标
        past_year_metrics = self._calculate_period_metrics(past_year_data)

        # 创建图表并转换为Base64
        nav_chart_base64 = self._create_nav_chart_base64(backtest_data)
        monthly_nav_chart_base64 = self._create_nav_chart_base64(past_month_data, title='Past Month NAV')
        yearly_nav_chart_base64 = self._create_nav_chart_base64(past_year_data, title='Past Year NAV')

        # 获取涨停跌停股票数据（如果可用）
        limit_stocks_info = ""
        if push_wave_data is not None and not push_wave_data.empty:
            latest_push_wave = push_wave_data.iloc[-1]
            limit_stocks_info = f"""
            <p><strong>Today&apos;s Limit-Up / Limit-Down Summary:</strong></p>
            <ul>
                <li>Limit-Up Ratio: {latest_push_wave.get('limit_up_ratio', 0):.4%}</li>
                <li>Limit-Down Ratio: {latest_push_wave.get('limit_down_ratio', 0):.4%}</li>
                <li>Push Wave Ratio: {latest_push_wave.get('push_wave_ratio', 0):.4%}</li>
            </ul>
            """

        # 创建HTML内容
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Push Wave Timing Strategy Report - Last Updated: {last_date}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    padding-bottom: 5px;
                    border-bottom: 1px solid #ecf0f1;
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 15px;
                    margin-bottom: 30px;
                }}
                .metrics-table th, .metrics-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: center;
                }}
                .metrics-table th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                .metrics-table tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .metrics-table tr:hover {{
                    background-color: #f1f1f1;
                }}
                .summary-box {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #3498db;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .return-positive {{
                    color: #27ae60;
                    font-weight: bold;
                }}
                .return-negative {{
                    color: #e74c3c;
                    font-weight: bold;
                }}
                .chart-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .footer {{
                    margin-top: 40px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 12px;
                    border-top: 1px solid #ecf0f1;
                    padding-top: 20px;
                }}
                .parameter-list {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                    gap: 10px;
                    margin: 20px 0;
                }}
                .parameter-item {{
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-radius: 4px;
                }}
                .parameter-label {{
                    font-weight: bold;
                    color: #2c3e50;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Push Wave Timing Strategy Report - Last Updated: {last_date}</h1>

                <div class="summary-box">
                    <h2>Strategy Summary</h2>
                    <div class="parameter-list">
                        <div class="parameter-item">
                            <span class="parameter-label">Current Position:</span> {today_position:.0f} ({"Fully Invested" if today_position == 1 else "Flat"})
                        </div>
                        <div class="parameter-item">
                            <span class="parameter-label">Today&apos;s Return:</span> 
                            <span class="{'return-positive' if today_return >= 0 else 'return-negative'}">
                                {today_return:.2%}
                            </span>
                        </div>
                        <div class="parameter-item">
                            <span class="parameter-label">Strategy NAV:</span> {backtest_data['strategy_nav'].iloc[-1]:,.2f}
                        </div>
                        <div class="parameter-item">
                            <span class="parameter-label">Benchmark NAV:</span> {backtest_data['benchmark_nav'].iloc[-1]:,.2f}
                        </div>
                    </div>
                </div>

                <h2>Today&apos;s Strategy Return Details</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Strategy</th>
                            <th>Benchmark</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Today&apos;s Return</td>
                            <td class="{'return-positive' if today_return >= 0 else 'return-negative'}">
                                {today_return:.2%}
                            </td>
                            <td class="{'return-positive' if backtest_data['return'].iloc[-1] >= 0 else 'return-negative'}">
                                {backtest_data['return'].iloc[-1]:.2%}
                            </td>
                        </tr>
                        <tr>
                            <td>Cumulative Return</td>
                            <td class="{'return-positive' if backtest_data['strategy_cum_return'].iloc[-1] >= 0 else 'return-negative'}">
                                {backtest_data['strategy_cum_return'].iloc[-1]:.2%}
                            </td>
                            <td class="{'return-positive' if backtest_data['benchmark_cum_return'].iloc[-1] >= 0 else 'return-negative'}">
                                {backtest_data['benchmark_cum_return'].iloc[-1]:.2%}
                            </td>
                        </tr>
                    </tbody>
                </table>

                {limit_stocks_info}

                <h2>Performance Metrics for the Past Month</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Strategy</th>
                            <th>Benchmark</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Annualized Return</td>
                            <td class="{'return-positive' if past_month_metrics.get('annual_return', 0) >= 0 else 'return-negative'}">
                                {past_month_metrics.get('annual_return', 0):.2%}
                            </td>
                            <td class="{'return-positive' if past_month_metrics.get('benchmark_annual', 0) >= 0 else 'return-negative'}">
                                {past_month_metrics.get('benchmark_annual', 0):.2%}
                            </td>
                        </tr>
                        <tr>
                            <td>Annualized Volatility</td>
                            <td>{past_month_metrics.get('annual_volatility', 0):.2%}</td>
                            <td>{past_month_metrics.get('benchmark_volatility', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td>{past_month_metrics.get('sharpe_ratio', 0):.3f}</td>
                            <td>{past_month_metrics.get('benchmark_sharpe', 0):.3f}</td>
                        </tr>
                        <tr>
                            <td>Max Drawdown</td>
                            <td class="return-negative">{past_month_metrics.get('max_drawdown', 0):.2%}</td>
                            <td class="return-negative">{past_month_metrics.get('benchmark_max_drawdown', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td>Win Rate</td>
                            <td>{past_month_metrics.get('win_rate', 0):.1%}</td>
                            <td>{past_month_metrics.get('benchmark_win_rate', 0):.1%}</td>
                        </tr>
                    </tbody>
                </table>

                <div class="chart-container">
                    <h3>Past Month NAV Curve</h3>
                    <img src="data:image/png;base64,{monthly_nav_chart_base64}" alt="Past Month NAV Curve">
                </div>

                <h2>Performance Metrics for the Past Year</h2>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Strategy</th>
                            <th>Benchmark</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Annualized Return</td>
                            <td class="{'return-positive' if past_year_metrics.get('annual_return', 0) >= 0 else 'return-negative'}">
                                {past_year_metrics.get('annual_return', 0):.2%}
                            </td>
                            <td class="{'return-positive' if past_year_metrics.get('benchmark_annual', 0) >= 0 else 'return-negative'}">
                                {past_year_metrics.get('benchmark_annual', 0):.2%}
                            </td>
                        </tr>
                        <tr>
                            <td>Annualized Volatility</td>
                            <td>{past_year_metrics.get('annual_volatility', 0):.2%}</td>
                            <td>{past_year_metrics.get('benchmark_volatility', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td>{past_year_metrics.get('sharpe_ratio', 0):.3f}</td>
                            <td>{past_year_metrics.get('benchmark_sharpe', 0):.3f}</td>
                        </tr>
                        <tr>
                            <td>Max Drawdown</td>
                            <td class="return-negative">{past_year_metrics.get('max_drawdown', 0):.2%}</td>
                            <td class="return-negative">{past_year_metrics.get('benchmark_max_drawdown', 0):.2%}</td>
                        </tr>
                        <tr>
                            <td>Win Rate</td>
                            <td>{past_year_metrics.get('win_rate', 0):.1%}</td>
                            <td>{past_year_metrics.get('benchmark_win_rate', 0):.1%}</td>
                        </tr>
                    </tbody>
                </table>

                <div class="chart-container">
                    <h3>Past Year NAV Curve</h3>
                    <img src="data:image/png;base64,{yearly_nav_chart_base64}" alt="Past Year NAV Curve">
                </div>

                <div class="chart-container">
                    <h3>Full NAV Curve</h3>
                    <img src="data:image/png;base64,{nav_chart_base64}" alt="Full NAV Curve">
                </div>

                <h2>Strategy Parameters</h2>
                <div class="parameter-list">
                    <div class="parameter-item">
                        <span class="parameter-label">Limit Threshold (mu):</span> {self.threshold * 100}%
                    </div>
                    <div class="parameter-item">
                        <span class="parameter-label">Short HMA Length:</span> {self.short_len} days
                    </div>
                    <div class="parameter-item">
                        <span class="parameter-label">Long HMA Length:</span> {self.long_len} days
                    </div>
                    <div class="parameter-item">
                        <span class="parameter-label">Signal Threshold:</span> {self.signal_threshold}
                    </div>
                    <div class="parameter-item">
                        <span class="parameter-label">Initial Capital:</span> {StrategyConfig.INITIAL_CAPITAL:,.0f}
                    </div>
                    <div class="parameter-item">
                        <span class="parameter-label">Transaction Cost:</span> {StrategyConfig.TRANSACTION_COST:.3%}
                    </div>
                    <div class="parameter-item">
                        <span class="parameter-label">Backtest Period:</span> {StrategyConfig.START_DATE} - {StrategyConfig.END_DATE}
                    </div>
                </div>

                <div class="footer">
                    <p>Report Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Push Wave Timing Model - Hull Moving Average Version</p>
                    <p>For reference only. Investing involves risk; make decisions with caution.</p>
                </div>
            </div>
        </body>
        </html>
        '''

        # 保存HTML文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML报告已生成: {filepath}")
        return filepath

    def _calculate_period_metrics(self, period_data: pd.DataFrame) -> Dict:
        """
        计算指定期间的绩效指标

        参数:
        ----------
        period_data : pd.DataFrame
            期间数据

        返回:
        ----------
        Dict
            绩效指标字典
        """
        if len(period_data) < 2:
            return {}

        # 策略年化收益率
        total_return = period_data['strategy_nav'].iloc[-1] / period_data['strategy_nav'].iloc[0] - 1
        years = len(period_data) / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return

        # 基准年化收益率
        benchmark_return = period_data['benchmark_nav'].iloc[-1] / period_data['benchmark_nav'].iloc[0] - 1
        benchmark_annual = (1 + benchmark_return) ** (1 / years) - 1 if years > 0 else benchmark_return

        # 策略年化波动率
        strategy_volatility = period_data['strategy_daily_return'].std() * np.sqrt(252) if len(period_data) > 0 else 0

        # 基准年化波动率
        benchmark_volatility = period_data['return'].std() * np.sqrt(252) if len(period_data) > 0 else 0

        # 夏普比率
        risk_free_rate = StrategyConfig.RISK_FREE_RATE
        sharpe_ratio = (annual_return - risk_free_rate) / strategy_volatility if strategy_volatility > 0 else 0
        benchmark_sharpe = (benchmark_annual - risk_free_rate) / benchmark_volatility if benchmark_volatility > 0 else 0

        # 最大回撤
        strategy_nav = period_data['strategy_nav']
        rolling_max = strategy_nav.expanding().max()
        drawdown = (strategy_nav - rolling_max) / rolling_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        benchmark_nav = period_data['benchmark_nav']
        benchmark_rolling_max = benchmark_nav.expanding().max()
        benchmark_drawdown = (benchmark_nav - benchmark_rolling_max) / benchmark_rolling_max
        benchmark_max_drawdown = benchmark_drawdown.min() if len(benchmark_drawdown) > 0 else 0

        # 胜率
        strategy_win_rate = (period_data['strategy_daily_return'] > 0).mean() if len(period_data) > 0 else 0
        benchmark_win_rate = (period_data['return'] > 0).mean() if len(period_data) > 0 else 0

        return {
            'annual_return': annual_return,
            'benchmark_annual': benchmark_annual,
            'annual_volatility': strategy_volatility,
            'benchmark_volatility': benchmark_volatility,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'win_rate': strategy_win_rate,
            'benchmark_win_rate': benchmark_win_rate,
            'total_return': total_return,
            'benchmark_return': benchmark_return
        }

    def _create_nav_chart_base64(self, data: pd.DataFrame, title: str = 'Strategy NAV Curve') -> str:
        """
        创建净值曲线图并返回Base64编码

        参数:
        ----------
        data : pd.DataFrame
            回测数据
        title : str
            图表标题

        返回:
        ----------
        str
            Base64编码的图像字符串
        """
        if len(data) < 2:
            return ""

        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            # 绘制策略净值曲线
            ax.plot(data.index, data['strategy_nav'], label='Strategy NAV', color='red', linewidth=2)

            # 绘制基准净值曲线
            ax.plot(data.index, data['benchmark_nav'], label='Benchmark NAV', color='blue', alpha=0.7, linewidth=1.5)

            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('NAV')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 格式化x轴日期
            if len(data) > 60:  # 如果数据点多，简化x轴标签
                ax.xaxis.set_major_locator(plt.MaxNLocator(8))

            plt.tight_layout()

            # 将图表转换为Base64
            buffer = io.BytesIO()
            canvas = FigureCanvas(fig)
            canvas.print_png(buffer)
            plt.close(fig)

            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return base64_str

        except Exception as e:
            print(f"创建图表时出错: {e}")
            return ""

    def run_full_analysis(self,
                          start_date: str = StrategyConfig.START_DATE,
                          end_date: str = StrategyConfig.END_DATE,
                          index_code: str = StrategyConfig.INDEX_CODE,
                          export_to_excel: bool = True,
                          export_to_html: bool = True,
                          export_limit_stocks: bool = True) -> Dict:
        """
        运行完整的分析流程
        """
        print("=" * 60)
        print("推波助澜择时模型 - 完整分析（Hull移动平均线版本）")
        print("=" * 60)
        print(f"参数设置: μ={self.threshold * 100}%, short_len={self.short_len}, "
              f"long_len={self.long_len}, threshold={self.signal_threshold}")
        print(f"时间范围: {start_date} 到 {end_date}")
        print(f"数据格式: Parquet列式存储")
        print(f"移动平均线: Hull移动平均线（HMA）")

        # 1. 计算推波助澜比率
        push_wave_data = self.calculate_push_wave_ratio(start_date, end_date, save_limit_stocks=export_limit_stocks)

        # 2. 计算Hull移动平均线
        ma_data = self.calculate_moving_averages(push_wave_data['push_wave_ratio'])

        # 3. 生成交易信号
        signals_df = self.generate_trading_signals(ma_data)

        # 4. 获取指数收益率数据
        index_data = self.data_manager.get_index_data(index_code, start_date, end_date)
        if index_data.empty:
            raise ValueError(f"无法获取指数{index_code}数据")

        # 5. 回测
        backtest_data = self.backtest_with_trading(signals_df, index_data['return'])

        # 6. 计算绩效指标
        metrics = self.calculate_performance_metrics(backtest_data)

        # 7. 打印交易摘要
        self.print_trading_summary()

        # 8. 可视化
        self.plot_results(backtest_data, ma_data, push_wave_data, signals_df)

        # 9. 导出到Excel
        if export_to_excel:
            excel_file = self.export_daily_positions_to_excel(
                backtest_data, ma_data, push_wave_data
            )
            print(f"\n✓ 分析结果已导出到Excel文件: {excel_file}")

        # 10. 生成HTML报告
        if export_to_html:
            html_file = self.generate_html_report(
                backtest_data, push_wave_data, ma_data
            )
            print(f"\n✓ HTML策略报告已生成: {html_file}")

        results = {
            'push_wave_data': push_wave_data,
            'ma_data': ma_data,
            'signals_df': signals_df,
            'backtest_data': backtest_data,
            'trade_records': self.trade_records,
            'daily_limit_stocks': self.daily_limit_stocks,
            'metrics': metrics,
            'parameters': {
                'threshold': self.threshold * 100,
                'short_len': self.short_len,
                'long_len': self.long_len,
                'signal_threshold': self.signal_threshold,
                'start_date': start_date,
                'end_date': end_date,
                'index_code': index_code
            }
        }

        return results


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序"""
    print("=" * 80)
    print("推波助澜择时模型 - Hull移动平均线版本")
    print("=" * 80)
    print(f"数据时间范围: {StrategyConfig.START_DATE} 到 {StrategyConfig.END_DATE}")
    print(f"数据存储格式: Parquet列式存储")
    print(f"移动平均线类型: Hull移动平均线（HMA）")
    print(f"涨跌停阈值: μ={StrategyConfig.THRESHOLD_MU}%")

    # 初始化tushare pro
    try:
        # 请替换为您的tushare token
        ts_token = TS_KEY
        ts.set_token(ts_token)
        pro = ts.pro_api()

        # 测试连接
        test_data = pro.trade_cal(exchange='', start_date='20240101', end_date='20240105')
        if test_data.empty:
            print("警告: tushare连接测试失败")
        else:
            print("tushare连接成功")
    except Exception as e:
        print(f"tushare初始化失败: {e}")
        print("\n请检查:")
        print("1. 是否设置了正确的tushare token")
        print("2. 网络连接是否正常")
        print("3. tushare账号是否有足够权限")
        return

    # 创建数据管理器
    data_manager = DataManager(pro)

    # 询问是否需要更新数据
    print("\n" + "=" * 80)
    print("数据管理选项:")
    print("=" * 80)
    print("1. 下载并缓存数据到Parquet格式（首次运行或更新数据）")
    print("2. 使用现有缓存数据运行策略")

    # choice = input("请选择(1/2，默认1): ").strip()
    choice = 1

    if choice == '2':
        print("使用现有缓存数据运行策略")
    else:
        # 批量下载数据到Parquet格式
        print("\n开始下载数据到本地Parquet文件...")
        data_manager.batch_download_daily_data(
            start_date=StrategyConfig.START_DATE,
            end_date=StrategyConfig.END_DATE,
            update_existing=False
        )

    # 运行推波助澜模型
    print("\n" + "=" * 80)
    print("运行推波助澜择时模型（Hull移动平均线版本）")
    print("=" * 80)

    model = PushWaveModel(data_manager=data_manager)

    try:
        results = model.run_full_analysis(
            start_date=StrategyConfig.START_DATE,
            end_date=StrategyConfig.END_DATE,
            index_code=StrategyConfig.INDEX_CODE,
            export_to_excel=True,
            export_to_html=True,
            export_limit_stocks=True
        )

        # 打印结果摘要
        print("\n" + "=" * 80)
        print("策略回测结果摘要")
        print("=" * 80)

        metrics = results['metrics']
        print(f"年化收益率: {metrics['年化收益率']:.2%}")
        print(f"基准年化收益率: {metrics['基准年化收益率']:.2%}")
        print(f"夏普比率: {metrics['夏普比率']:.3f}")
        print(f"最大回撤: {metrics['最大回撤']:.2%}")
        print(f"胜率: {metrics['胜率']:.1%}")
        print(f"总交易次数: {int(metrics['总交易次数'])}次")
        print(f"平均多头持有期: {metrics['平均多头持有期(天)']:.1f}天")
        print(f"总收益率: {metrics['总收益率']:.2%}")

    except Exception as e:
        print(f"策略运行失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("程序执行完成！")
    print("=" * 80)


# ============================================================================
# 运行程序
# ============================================================================

if __name__ == "__main__":
    # 检查依赖库
    try:
        import pyarrow

        print("PyArrow库已安装，支持Parquet格式")
    except ImportError:
        print("警告: PyArrow库未安装，将无法使用Parquet格式")
        print("请运行: pip install pyarrow")

    # 检查base64模块
    try:
        import base64
        import io
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

        print("HTML报告生成所需库已安装")
    except ImportError as e:
        print(f"警告: HTML报告生成所需库未安装: {e}")
        print("请运行: pip install matplotlib")

    # 运行主程序
    main()

    # 发送html
    # HTML_PATH = "output/strategy_dashboard.html"
    # try:
    #     with open(HTML_PATH, "r", encoding="utf-8") as f:
    #         HTML_BODY = f.read()
    # except Exception:
    #     HTML_BODY = "<p>Please find the attached file.</p>"
    # # HTML_BODY = "<p>Please find the attached file.</p>"
    # for re in pw.RECIPIENTS:
    #     print(f"Sending {HTML_PATH} to {re}")
    #     email_sender_v2.send_html_email_with_attachment(
    #         smtp_server="smtp.gmail.com",
    #         smtp_port=587,
    #         sender_email=pw.SENDER_EMAIL,    # your gmail
    #         password=pw.google_email_app_password,  # your gmail app password
    #         receiver_email=re,  # recipient email
    #         subject="Whitney George Daily Backtest Report",
    #         html_body=HTML_BODY,
    #         attachment_path=HTML_PATH
    #     )
