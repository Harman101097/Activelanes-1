#!/usr/bin/env python3

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from scipy.optimize import minimize
from scipy import stats
import warnings
import os
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    PLOTTING_AVAILABLE = True
    print("Matplotlib and Seaborn available - Charts enabled")
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn not available - Charts disabled")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

class CADBufferCalculator:
    def __init__(self):
        self.base_url = "https://www.bankofcanada.ca/valet/observations/FXUSDCAD"
        
        # Set plotting style if available
        if PLOTTING_AVAILABLE:
            plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
            sns.set_palette("husl")
        
        self.data_periods = {
            1: {
                'original_p1_months': 3, 'original_p2_months': 1, 'original_vol_periods': 12, 'original_vol_type': 'monthly',
                'egarch_years': 1, 'historical_years': 1, 'var_years': 2, 'arima_months': 6,
                'arima_short_months': 1, 'arima_medium_months': 4, 'forecast_days': 21
            },
            3: {
                'original_p1_months': 6, 'original_p2_months': 2, 'original_vol_periods': 12, 'original_vol_type': 'quarterly',
                'egarch_years': 2, 'historical_years': 2, 'var_years': 3, 'arima_months': 15,
                'arima_short_months': 3, 'arima_medium_months': 12, 'forecast_days': 63
            },
            6: {
                'original_p1_months': 12, 'original_p2_months': 4, 'original_vol_periods': 12, 'original_vol_type': 'biannual',
                'egarch_years': 3, 'historical_years': 3, 'var_years': 5, 'arima_months': 30,
                'arima_short_months': 6, 'arima_medium_months': 24, 'forecast_days': 126
            },
            12: {
                'original_p1_months': 24, 'original_p2_months': 12, 'original_vol_periods': 12, 'original_vol_type': 'annual',
                'egarch_years': 12, 'historical_years': 6, 'var_years': 10, 'arima_months': 48,
                'arima_short_months': 12, 'arima_medium_months': 36, 'forecast_days': 252
            }
        }

    def diagnose_data_issue(self):
        """
        Diagnostic function to identify why results are identical
        """
        print("="*60)
        print("DIAGNOSTIC MODE - CHECKING FOR ISSUES")
        print("="*60)
        
        # Fetch fresh data
        df = self.fetch_cad_data(days_back=60)
        
        if df is None or df.empty:
            print("ERROR: No data fetched!")
            return
        
        # Check data freshness
        latest_date = df['date'].iloc[-1]
        current_date = pd.Timestamp.now()
        data_age_days = (current_date - latest_date).days
        
        print(f"Current system time: {current_date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Latest data date: {latest_date.strftime('%Y-%m-%d')}")
        print(f"Data age: {data_age_days} days")
        
        if data_age_days > 3:
            print(f"WARNING: Data is {data_age_days} days old!")
            print("This could explain identical results.")
        
        # Check recent rate changes
        print(f"\nRecent rate changes (last 10 data points):")
        recent_data = df.tail(10)[['date', 'rate']].copy()
        recent_data['change'] = recent_data['rate'].diff()
        recent_data['change_pct'] = recent_data['rate'].pct_change() * 100
        
        for _, row in recent_data.iterrows():
            change_str = f"{row['change']:+.4f}" if pd.notna(row['change']) else "N/A"
            pct_str = f"({row['change_pct']:+.2f}%)" if pd.notna(row['change_pct']) else ""
            print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['rate']:.4f} {change_str} {pct_str}")
        
        # Check if current rate actually changed
        current_rate = df['rate'].iloc[-1]
        prev_rate = df['rate'].iloc[-2] if len(df) > 1 else current_rate
        rate_change = current_rate - prev_rate
        
        print(f"\nCurrent rate: {current_rate:.4f}")
        print(f"Previous rate: {prev_rate:.4f}")
        print(f"Change: {rate_change:+.4f}")
        
        if abs(rate_change) < 0.0001:
            print("Rate hasn't changed - this explains identical buffer results!")
        
        print("\n" + "="*60)
        
    def fetch_cad_data(self, days_back=450):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {'start_date': start_date, 'end_date': end_date, 'order_dir': 'asc'}
        
        try:
            print(f"Fetching data from {start_date} to {end_date}...")
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            rates = []
            for obs in data['observations']:
                if obs['FXUSDCAD']['v'] is not None:
                    rates.append({'date': pd.to_datetime(obs['d']), 'rate': float(obs['FXUSDCAD']['v'])})
            
            df = pd.DataFrame(rates)
            df = df.sort_values('date').reset_index(drop=True)
            
            # Add debugging info
            print(f"Fetched {len(df)} observations from Bank of Canada")
            if len(df) > 0:
                latest_date = df['date'].iloc[-1]
                current_date = pd.Timestamp.now()
                data_age = (current_date.date() - latest_date.date()).days
                
                print(f"Latest data: {latest_date.strftime('%Y-%m-%d')} - Rate: {df['rate'].iloc[-1]:.4f}")
                print(f"Data age: {data_age} days")
                
                if data_age > 3:
                    print(f"WARNING: Data is {data_age} days old!")
                
                if len(df) > 1:
                    rate_change = df['rate'].iloc[-1] - df['rate'].iloc[-2]
                    print(f"Rate change from previous: {rate_change:+.4f}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def remove_outliers(self, data, method='iqr', factor=1.5, percentile_lower=2.5, percentile_upper=97.5):
        """
        Remove outliers from data using specified method.
        """
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
            
        elif method == 'percentile':
            lower_bound = data.quantile(percentile_lower/100)
            upper_bound = data.quantile(percentile_upper/100)
            cleaned_data = data[(data >= lower_bound) & (data <= upper_bound)]
            
        else:
            print(f"   Warning: Unknown outlier method '{method}', using IQR")
            return self.remove_outliers(data, method='iqr', factor=factor)
        
        outliers_removed = len(data) - len(cleaned_data)
        removal_percentage = (outliers_removed / len(data)) * 100
        print(f"   Outlier removal ({method}): {outliers_removed} outliers removed ({removal_percentage:.1f}%)")
        
        return cleaned_data

    def calculate_original_buffer(self, df, scenario, volatility_percentile, time_horizon):
        current_rate = df['rate'].iloc[-1]
        latest_data_date = df['date'].iloc[-1]
        periods = self.data_periods[time_horizon]
        
        if periods['original_vol_type'] == 'monthly':
            df_temp = df.copy()
            df_temp['period'] = df_temp['date'].dt.to_period('M')
            latest_period = df_temp['period'].max()
            lookback_periods = latest_period - (periods['original_vol_periods'] - 1)
            period_data = df_temp[df_temp['period'] >= lookback_periods]
            period_spreads = period_data.groupby('period')['rate'].agg(['min', 'max'])
        elif periods['original_vol_type'] == 'quarterly':
            df_temp = df.copy()
            df_temp['period'] = df_temp['date'].dt.to_period('Q')
            latest_period = df_temp['period'].max()
            lookback_periods = latest_period - (periods['original_vol_periods'] - 1)
            period_data = df_temp[df_temp['period'] >= lookback_periods]
            period_spreads = period_data.groupby('period')['rate'].agg(['min', 'max'])
        elif periods['original_vol_type'] == 'biannual':
            df_temp = df.copy()
            df_temp['half_year'] = df_temp['date'].dt.year.astype(str) + '-H' + ((df_temp['date'].dt.month - 1) // 6 + 1).astype(str)
            period_data = df_temp.groupby('half_year').tail(len(df_temp)).groupby('half_year')['rate'].agg(['min', 'max']).tail(periods['original_vol_periods'])
            period_spreads = period_data
        else:
            df_temp = df.copy()
            df_temp['period'] = df_temp['date'].dt.to_period('Y')
            latest_period = df_temp['period'].max()
            lookback_periods = latest_period - (periods['original_vol_periods'] - 1)
            period_data = df_temp[df_temp['period'] >= lookback_periods]
            period_spreads = period_data.groupby('period')['rate'].agg(['min', 'max'])
        
        period_spreads['spread'] = period_spreads['max'] - period_spreads['min']
        spreads_list = period_spreads['spread'].tolist()
        
        percentiles = {25: np.percentile(spreads_list, 25), 50: np.percentile(spreads_list, 50), 75: np.percentile(spreads_list, 75), 90: np.percentile(spreads_list, 90)}
        vol = percentiles[volatility_percentile]
        
        p1_days_ago = latest_data_date - pd.Timedelta(days=periods['original_p1_months'] * 30)
        p2_days_ago = latest_data_date - pd.Timedelta(days=periods['original_p2_months'] * 30)
        
        if scenario.lower() == 'depreciating':
            recent_p1 = df[df['date'] >= p1_days_ago]
            p1 = len(recent_p1[recent_p1['rate'] > current_rate]) / len(recent_p1) if len(recent_p1) > 0 else 0.0
            
            recent_p2 = df[df['date'] >= p2_days_ago].copy()
            recent_p2['daily_change'] = recent_p2['rate'].diff()
            p2 = len(recent_p2[recent_p2['daily_change'] > 0]) / len(recent_p2.dropna()) if len(recent_p2.dropna()) > 0 else 0.0
            
            buffer_price = (p1 * vol) + (p2 * vol)
            final_rate = current_rate + buffer_price
        else:
            recent_p1 = df[df['date'] >= p1_days_ago]
            p1 = len(recent_p1[recent_p1['rate'] < current_rate]) / len(recent_p1) if len(recent_p1) > 0 else 0.0
            
            recent_p2 = df[df['date'] >= p2_days_ago].copy()
            recent_p2['daily_change'] = recent_p2['rate'].diff()
            p2 = len(recent_p2[recent_p2['daily_change'] < 0]) / len(recent_p2.dropna()) if len(recent_p2.dropna()) > 0 else 0.0
            
            buffer_price = (p1 * vol) + (p2 * vol)
            final_rate = current_rate - buffer_price
        
        buffer_percentage = (buffer_price / current_rate) * 100
        print(f"   P1={p1:.3f}, P2={p2:.3f}, Buffer=${buffer_price:.4f}")
        
        return {
            'method': 'Original Formula',
            'buffer_price': buffer_price,
            'buffer_percentage': buffer_percentage,
            'final_rate': final_rate,
            'p1': p1, 'p2': p2, 'vol': vol,
            'time_horizon': f"{time_horizon} month(s)",
            'data_periods_used': f"P1:{periods['original_p1_months']}m, P2:{periods['original_p2_months']}m"
        }

    def fit_egarch_model(self, returns):
        def egarch_likelihood(params, returns):
            omega, alpha, beta, gamma = params
            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(returns)
            
            log_likelihood = 0
            for t in range(1, n):
                epsilon_t_1 = returns.iloc[t-1] / np.sqrt(max(sigma2[t-1], 1e-8))
                log_sigma2_t = (omega + alpha * abs(epsilon_t_1) + gamma * epsilon_t_1 + beta * np.log(max(sigma2[t-1], 1e-8)))
                sigma2[t] = np.exp(log_sigma2_t)
                sigma2[t] = max(min(sigma2[t], 1.0), 1e-8)
                
                log_likelihood -= 0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + (returns.iloc[t]**2) / sigma2[t])
            return -log_likelihood
        
        initial_params = [0.01, 0.1, 0.8, 0.05]
        bounds = [(-0.5, 0.5), (0.01, 0.5), (0.1, 0.95), (-0.5, 0.5)]
        
        try:
            result = minimize(egarch_likelihood, initial_params, args=(returns,), bounds=bounds, method='L-BFGS-B', options={'maxiter': 500})
            
            if result.success:
                omega, alpha, beta, gamma = result.x
                print(f"   EGARCH fitted: ω={omega:.4f}, α={alpha:.4f}, β={beta:.4f}, γ={gamma:.4f}")
                return result.x, True
            else:
                print("   EGARCH fitting failed, using enhanced volatility")
                return None, False
        except Exception as e:
            print(f"   EGARCH error: {e}")
            return None, False
    
    def predict_egarch_volatility(self, params, returns, forecast_days):
        omega, alpha, beta, gamma = params
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            epsilon_t_1 = returns.iloc[t-1] / np.sqrt(max(sigma2[t-1], 1e-8))
            log_sigma2_t = (omega + alpha * abs(epsilon_t_1) + gamma * epsilon_t_1 + beta * np.log(max(sigma2[t-1], 1e-8)))
            sigma2[t] = np.exp(log_sigma2_t)
            sigma2[t] = max(min(sigma2[t], 1.0), 1e-8)
        
        last_epsilon = returns.iloc[-1] / np.sqrt(max(sigma2[-1], 1e-8))
        predicted_log_sigma2 = (omega + alpha * abs(last_epsilon) + gamma * last_epsilon + beta * np.log(max(sigma2[-1], 1e-8)))
        
        predicted_daily_vol = np.sqrt(np.exp(predicted_log_sigma2))
        predicted_daily_vol = max(min(predicted_daily_vol, 0.05), 0.001)
        
        period_vol = predicted_daily_vol * np.sqrt(forecast_days)
        print(f"   EGARCH daily vol: {predicted_daily_vol:.4f}, period vol: {period_vol:.4f}")
        
        return period_vol

    def calculate_egarch_buffer(self, df, scenario, volatility_percentile, time_horizon):
        current_rate = df['rate'].iloc[-1]
        latest_data_date = df['date'].iloc[-1]
        periods = self.data_periods[time_horizon]
        
        years_back = periods['egarch_years']
        years_ago = latest_data_date - pd.Timedelta(days=years_back * 365)
        yearly_data = df[df['date'] >= years_ago]
        
        if len(yearly_data) < 50:
            print(f"   Not enough data for EGARCH model ({len(yearly_data)} days)")
            return None
        
        print(f"   EGARCH: Using {len(yearly_data)} days ({years_back} years)")
        
        yearly_data_copy = yearly_data.copy()
        yearly_data_copy['log_returns'] = np.log(yearly_data_copy['rate'] / yearly_data_copy['rate'].shift(1))
        returns = yearly_data_copy['log_returns'].dropna()
        
        egarch_params, success = self.fit_egarch_model(returns)
        
        if success:
            monthly_vol = self.predict_egarch_volatility(egarch_params, returns, periods['forecast_days'])
        else:
            daily_vol = returns.std()
            rolling_vol = returns.rolling(30).std().mean() if len(returns) > 30 else daily_vol
            realized_vol = np.sqrt(np.mean(returns**2))
            
            base_vol = (0.4 * daily_vol + 0.3 * rolling_vol + 0.3 * realized_vol)
            monthly_vol = base_vol * np.sqrt(periods['forecast_days'])
            print(f"   Enhanced volatility used: {monthly_vol:.4f}")
        
        percentile_multipliers = {25: 0.25, 50: 0.50, 75: 0.75, 90: 0.90}
        buffer_price = percentile_multipliers[volatility_percentile] * monthly_vol * current_rate
        
        if scenario.lower() == 'depreciating':
            final_rate = current_rate + buffer_price
        else:
            final_rate = current_rate - buffer_price
        
        buffer_percentage = (buffer_price / current_rate) * 100
        
        return {
            'method': 'EGARCH + Monte Carlo',
            'buffer_price': buffer_price,
            'buffer_percentage': buffer_percentage,
            'final_rate': final_rate,
            'monthly_volatility': monthly_vol,
            'egarch_fitted': success,
            'time_horizon': f"{time_horizon} month(s)",
            'data_periods_used': f"{years_back} years of data"
        }

    def calculate_historical_buffer(self, df, scenario, volatility_percentile, time_horizon):
        current_rate = df['rate'].iloc[-1]
        latest_data_date = df['date'].iloc[-1]
        periods = self.data_periods[time_horizon]
        
        years_back = periods['historical_years']
        years_ago = latest_data_date - pd.Timedelta(days=years_back * 365)
        yearly_data = df[df['date'] >= years_ago].copy()
        
        if len(yearly_data) < 50:
            print(f"   Not enough historical data ({len(yearly_data)} days)")
            return None
        
        print(f"   Historical: Using {len(yearly_data)} days ({years_back} years)")
        
        yearly_data['daily_return'] = yearly_data['rate'].pct_change()
        yearly_data['log_return'] = np.log(yearly_data['rate'] / yearly_data['rate'].shift(1))
        
        daily_returns = yearly_data['daily_return'].dropna()
        log_returns = yearly_data['log_return'].dropna()
        
        forecast_days = periods['forecast_days']
        
        # Calculate basic volatilities (same as before)
        simple_vol = daily_returns.std() * np.sqrt(forecast_days)
        log_vol = log_returns.std() * np.sqrt(forecast_days)
        
        if len(log_returns) >= forecast_days:
            realized_vol = np.sqrt((log_returns**2).rolling(forecast_days).mean().mean()) * np.sqrt(forecast_days)
            garch_like_vol = np.sqrt((log_returns**2).ewm(span=30).mean().iloc[-1]) * np.sqrt(forecast_days)
        else:
            realized_vol = simple_vol
            garch_like_vol = simple_vol
        
        # DIRECTIONAL VOLATILITY ANALYSIS (67/33 weighting)
        # Separate returns by direction
        up_returns = daily_returns[daily_returns > 0]
        down_returns = daily_returns[daily_returns < 0]
        
        print(f"   Directional analysis: {len(up_returns)} up days, {len(down_returns)} down days")
        
        if scenario.lower() == 'depreciating':
            # Focus on upward movements (CAD weakening = USD/CAD rate increases)
            if len(up_returns) >= 10:
                directional_vol = up_returns.std() * np.sqrt(forecast_days)
                directional_type = "upward"
            else:
                directional_vol = simple_vol
                directional_type = "insufficient up data, using overall"
            
            # Weight more toward volatilities that capture upward risk
            vol_weights = [0.25, 0.25, 0.15, 0.35]  # More weight on GARCH-like
            
        else:  # appreciating scenario
            # Focus on downward movements (CAD strengthening = USD/CAD rate decreases)
            if len(down_returns) >= 10:
                directional_vol = abs(down_returns.std()) * np.sqrt(forecast_days)
                directional_type = "downward"
            else:
                directional_vol = simple_vol
                directional_type = "insufficient down data, using overall"
            
            # Weight more toward volatilities that capture downward risk
            vol_weights = [0.25, 0.25, 0.35, 0.15]  # More weight on realized vol
        
        print(f"   Using {directional_type} volatility: {directional_vol:.4f}")
        if len(up_returns) > 0:
            print(f"   Up moves avg: {up_returns.mean():.4f}, std: {up_returns.std():.4f}")
        if len(down_returns) > 0:
            print(f"   Down moves avg: {down_returns.mean():.4f}, std: {down_returns.std():.4f}")
        
        # Calculate weighted volatility with directional component
        base_weighted_vol = (vol_weights[0] * simple_vol + 
                            vol_weights[1] * log_vol + 
                            vol_weights[2] * realized_vol + 
                            vol_weights[3] * garch_like_vol)
        
        # Blend base volatility with directional volatility (67/33 weighting)
        base_weight = 0.67  # 67% base volatility
        directional_weight = 0.33  # 33% directional volatility
        weighted_vol = base_weight * base_weighted_vol + directional_weight * directional_vol
        
        print(f"   Base vol: {base_weighted_vol:.4f}, Directional vol: {directional_vol:.4f}")
        print(f"   Final weighted vol (67/33): {weighted_vol:.4f}")
        
        # Apply percentile multiplier
        percentile_multipliers = {25: 0.25, 50: 0.50, 75: 0.75, 90: 0.90}
        multiplier = percentile_multipliers[volatility_percentile]
        
        buffer_price = multiplier * weighted_vol * current_rate
        
        if scenario.lower() == 'depreciating':
            final_rate = current_rate + buffer_price
        else:
            final_rate = current_rate - buffer_price
        
        buffer_percentage = (buffer_price / current_rate) * 100
        
        return {
            'method': 'Enhanced Historical Averages (67/33 Directional)',
            'buffer_price': buffer_price,
            'buffer_percentage': buffer_percentage,
            'final_rate': final_rate,
            'weighted_vol': weighted_vol,
            'directional_vol': directional_vol,
            'directional_type': directional_type,
            'up_days': len(up_returns),
            'down_days': len(down_returns),
            'up_avg': up_returns.mean() if len(up_returns) > 0 else 0,
            'down_avg': down_returns.mean() if len(down_returns) > 0 else 0,
            'base_weight': base_weight,
            'directional_weight': directional_weight,
            'time_horizon': f"{time_horizon} month(s)",
            'data_periods_used': f"{years_back} years of data with 67/33 directional analysis"
        }

    def calculate_var_buffer(self, df, scenario, volatility_percentile, time_horizon):
        current_rate = df['rate'].iloc[-1]
        latest_data_date = df['date'].iloc[-1]
        periods = self.data_periods[time_horizon]
        
        years_back = periods['var_years']
        years_ago = latest_data_date - pd.Timedelta(days=years_back * 365)
        var_data = df[df['date'] >= years_ago].copy()
        
        if len(var_data) < 100:
            print(f"   Not enough data for VaR model ({len(var_data)} days)")
            return None
        
        print(f"   VaR Model: Using {len(var_data)} days ({years_back} years)")
        
        var_data['log_return'] = np.log(var_data['rate'] / var_data['rate'].shift(1))
        returns = var_data['log_return'].dropna()
        
        if len(returns) < 50:
            print("   Insufficient return data for VaR")
            return None
        
        returns_clean = self.remove_outliers(returns, method='iqr', factor=1.5)
        outliers_removed = len(returns) - len(returns_clean)
        print(f"   Removed {outliers_removed} outliers ({outliers_removed/len(returns)*100:.1f}%)")
        
        confidence_level = volatility_percentile / 100.0
        
        if scenario.lower() == 'depreciating':
            positive_returns = returns_clean[returns_clean > 0]
            
            if len(positive_returns) == 0:
                print("   No positive returns found for depreciating scenario")
                return None
            
            target_percentile = confidence_level * 100
            var_return = np.percentile(positive_returns, target_percentile)
            
            scaled_return = var_return * np.sqrt(periods['forecast_days'])
            predicted_rate = current_rate * np.exp(scaled_return)
            
            buffer_price = predicted_rate - current_rate
            final_rate = predicted_rate
            
            interpretation = f"{(1-confidence_level)*100:.0f}% chance CAD depreciates beyond {predicted_rate:.4f}"
            
        else:
            negative_returns = returns_clean[returns_clean < 0]
            
            if len(negative_returns) == 0:
                print("   No negative returns found for appreciating scenario")
                return None
            
            target_percentile = (1 - confidence_level) * 100
            var_return = np.percentile(negative_returns, target_percentile)
            
            scaled_return = var_return * np.sqrt(periods['forecast_days'])
            predicted_rate = current_rate * np.exp(scaled_return)
            
            buffer_price = current_rate - predicted_rate
            final_rate = predicted_rate
            
            interpretation = f"{(1-confidence_level)*100:.0f}% chance CAD appreciates beyond {predicted_rate:.4f}"
        
        buffer_percentage = (buffer_price / current_rate) * 100
        
        print(f"   VaR Prediction: Rate -> {predicted_rate:.4f}")
        print(f"   {interpretation}")
        
        return {
            'method': 'Value at Risk (Directional)',
            'buffer_price': buffer_price,
            'buffer_percentage': buffer_percentage,
            'final_rate': final_rate,
            'predicted_rate': predicted_rate,
            'confidence_level': confidence_level,
            'outliers_removed': outliers_removed,
            'interpretation': interpretation,
            'time_horizon': f"{time_horizon} month(s)",
            'data_periods_used': f"{years_back} years ({outliers_removed} outliers removed)"
        }

    def calculate_arima_buffer(self, df, scenario, volatility_percentile, time_horizon):
        current_rate = df['rate'].iloc[-1]
        latest_data_date = df['date'].iloc[-1]
        periods = self.data_periods[time_horizon]
        
        months_back = periods['arima_months']
        months_ago = latest_data_date - pd.Timedelta(days=months_back * 30)
        recent_data = df[df['date'] >= months_ago].copy()
        
        if len(recent_data) < 30:
            print(f"   Not enough data for ARIMA model ({len(recent_data)} days)")
            return None
        
        print(f"   ARIMA Model: Using {len(recent_data)} days ({months_back} months)")
        
        rates = recent_data['rate'].values
        
        if len(rates) < 20:
            print("   Insufficient rate data for ARIMA")
            return None
        
        arima_fitted = False
        forecast_rate = current_rate
        forecast_volatility = 0.01
        best_order = None
        
        if STATSMODELS_AVAILABLE:
            try:
                print("   ARIMA fitting process starting...")
                adf_result = adfuller(rates, autolag='AIC')
                is_stationary = adf_result[1] < 0.05
                print(f"   ADF Test p-value: {adf_result[1]:.4f}, Stationary: {is_stationary}")
                
                d_order = 0
                test_data = rates.copy()
                max_d = 2
                
                for d in range(max_d + 1):
                    if d == 0:
                        adf_stat, adf_p = adfuller(test_data, autolag='AIC')[:2]
                    else:
                        diff_data = np.diff(test_data, n=d)
                        if len(diff_data) > 10:
                            adf_stat, adf_p = adfuller(diff_data, autolag='AIC')[:2]
                        else:
                            break
                    
                    if adf_p < 0.05:
                        d_order = d
                        print(f"   Series is stationary after {d} differences (p-value: {adf_p:.4f})")
                        break
                
                if adf_p >= 0.05:
                    d_order = 1
                    print(f"   Using first difference as fallback (d={d_order})")
                
                best_aic = np.inf
                best_model = None
                
                max_p = min(3, len(rates) // 15)
                max_q = min(3, len(rates) // 15)
                
                print(f"   Searching ARIMA orders: p≤{max_p}, d={d_order}, q≤{max_q}")
                models_tried = 0
                models_succeeded = 0
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    for p in range(max_p + 1):
                        for q in range(max_q + 1):
                            models_tried += 1
                            try:
                                order = (p, d_order, q)
                                model = ARIMA(rates, order=order)
                                fitted_model = model.fit(method_kwargs={'warn_convergence': False})
                                
                                if fitted_model.aic < best_aic and np.isfinite(fitted_model.aic):
                                    best_aic = fitted_model.aic
                                    best_model = fitted_model
                                    best_order = order
                                    models_succeeded += 1
                                    
                            except Exception as e:
                                continue
                
                print(f"   Tried {models_tried} ARIMA configurations, {models_succeeded} succeeded")
                
                if best_model is not None:
                    print(f"   Best model: ARIMA{best_order} with AIC={best_aic:.2f}")
                    forecast_steps = min(periods['forecast_days'], 30)
                    
                    forecast_result = best_model.forecast(steps=forecast_steps)
                    if hasattr(forecast_result, 'iloc'):
                        forecast_rate = forecast_result.iloc[-1]
                    else:
                        forecast_rate = forecast_result[-1] if isinstance(forecast_result, (list, np.ndarray)) else forecast_result
                    
                    forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()
                    
                    if len(forecast_ci) > 0:
                        if hasattr(forecast_ci, 'iloc'):
                            forecast_range = forecast_ci.iloc[-1, 1] - forecast_ci.iloc[-1, 0]
                        else:
                            forecast_range = forecast_ci[-1, 1] - forecast_ci[-1, 0]
                        forecast_volatility = forecast_range / 4
                    else:
                        forecast_volatility = np.std(best_model.resid) if hasattr(best_model, 'resid') else 0.01
                    
                    arima_fitted = True
                    print(f"   ARIMA SUCCESSFULLY FITTED: {best_order}, AIC={best_aic:.2f}")
                    print(f"   Forecast rate: {forecast_rate:.4f}, Volatility: {forecast_volatility:.4f}")
                else:
                    print(f"   ARIMA FITTING FAILED: No valid models found")
                    
            except Exception as e:
                print(f"   ARIMA fitting error: {e}")
                print("   Falling back to enhanced trend method")
        else:
            print("   STATSMODELS not available, using fallback")
        
        if not arima_fitted:
            print("   Using fallback: Enhanced trend with mandatory outlier removal for medium-term")
            if len(rates) >= 10:
                short_window_days = periods['arima_short_months'] * 30
                medium_window_days = periods['arima_medium_months'] * 30
                
                short_window = min(short_window_days, len(rates) // 3)
                medium_window = min(medium_window_days, len(rates) // 2)
                
                print(f"   Using short window: {short_window} days, medium window: {medium_window} days")
                
                short_rates = rates[-int(short_window):] if short_window > 0 else rates[-5:]
                short_trend = (short_rates[-1] - short_rates[0]) / (len(short_rates) - 1) if len(short_rates) > 1 else 0
                
                medium_rates_raw = rates[-int(medium_window):] if medium_window > 0 else rates[-10:]
                medium_rates_series = pd.Series(medium_rates_raw)
                
                print("   MANDATORY outlier removal for medium-term trend:")
                medium_rates_clean = self.remove_outliers(medium_rates_series, 
                                                        method='percentile', 
                                                        percentile_lower=2.5, 
                                                        percentile_upper=97.5)
                
                if len(medium_rates_clean) == len(medium_rates_series):
                    print("   WARNING: No outliers were removed - applying IQR as backup")
                    medium_rates_clean = self.remove_outliers(medium_rates_series, method='iqr', factor=1.5)
                
                if len(medium_rates_clean) > 1:
                    medium_clean_array = medium_rates_clean.values
                    medium_trend = (medium_clean_array[-1] - medium_clean_array[0]) / (len(medium_clean_array) - 1)
                else:
                    print("   ERROR: Too few data points after outlier removal")
                    medium_trend = 0
                
                combined_trend = 0.5 * short_trend + 0.5 * medium_trend
                
                forecast_rate = current_rate + combined_trend * periods['forecast_days']
                
                rate_changes = np.diff(rates)
                forecast_volatility = np.std(rate_changes) * np.sqrt(periods['forecast_days']) if len(rate_changes) > 0 else 0.01
                
                print(f"   Short trend (with outliers): {short_trend:.6f}/day")
                print(f"   Medium trend (outliers removed): {medium_trend:.6f}/day")
                print(f"   Combined trend (0.5/0.5): {combined_trend:.6f}/day")
                
            else:
                forecast_rate = current_rate
                forecast_volatility = 0.01
                combined_trend = 0
            
            best_order = f"Trend Fallback (Short:{periods['arima_short_months']}m+outliers, Medium:{periods['arima_medium_months']}m-outliers_MANDATORY)"
        
        percentile_multipliers = {25: 0.67, 50: 1.0, 75: 1.35, 90: 1.65}
        multiplier = percentile_multipliers[volatility_percentile]
        
        rate_change = forecast_rate - current_rate
        forecast_direction = "up" if rate_change > 0 else "down"
        
        trend_component = abs(rate_change)
        volatility_component = multiplier * forecast_volatility
        
        if scenario.lower() == 'depreciating':
            if rate_change > 0:
                scenario_agreement = "agrees"
                arima_buffer = 0.5 * trend_component + 0.5 * volatility_component
                final_rate = current_rate + arima_buffer
            else:
                scenario_agreement = "disagrees"
                arima_buffer = 0.8 * volatility_component + 0.2 * trend_component
                final_rate = current_rate + arima_buffer
        else:
            if rate_change < 0:
                scenario_agreement = "agrees"
                arima_buffer = 0.5 * trend_component + 0.5 * volatility_component
                final_rate = current_rate - arima_buffer
            else:
                scenario_agreement = "disagrees"
                arima_buffer = 0.8 * volatility_component + 0.2 * trend_component
                final_rate = current_rate + arima_buffer
        
        buffer_percentage = (arima_buffer / current_rate) * 100
        
        print(f"   Rate change: {rate_change:+.4f} (forecast {forecast_direction})")
        print(f"   Scenario agreement: ARIMA {scenario_agreement} with {scenario} expectation")
        print(f"   ARIMA Buffer: Trend={trend_component:.4f} + Vol={volatility_component:.4f} = {arima_buffer:.4f}")
        
        return {
            'method': 'ARIMA Forecasting (Mandatory Outlier Removal)',
            'buffer_price': arima_buffer,
            'buffer_percentage': buffer_percentage,
            'final_rate': final_rate,
            'forecast_rate': forecast_rate,
            'rate_change': rate_change,
            'forecast_direction': forecast_direction,
            'scenario_agreement': scenario_agreement,
            'trend_component': trend_component,
            'volatility_component': volatility_component,
            'best_order': best_order,
            'arima_fitted': arima_fitted,
            'short_window': periods['arima_short_months'],
            'medium_window': periods['arima_medium_months'],
            'time_horizon': f"{time_horizon} month(s)",
            'data_periods_used': f"{months_back} months total, mandatory outlier removal for medium-term"
        }

    def calculate_all_methods(self, scenario, volatility_percentile, time_horizon):
        print("Fetching data from Bank of Canada API...")
        
        max_years = max([
            self.data_periods[time_horizon]['egarch_years'],
            self.data_periods[time_horizon]['historical_years'],
            self.data_periods[time_horizon]['var_years'],
            self.data_periods[time_horizon]['arima_months'] / 12,
            self.data_periods[time_horizon]['original_p1_months'] / 12
        ])
        days_back = int(max_years * 365) + 100
        
        df = self.fetch_cad_data(days_back=days_back)
        
        if df is None or df.empty:
            return {"error": "Failed to fetch data"}
        
        current_rate = df['rate'].iloc[-1]
        current_date = df['date'].iloc[-1].strftime('%Y-%m-%d')
        
        print(f"Current USD/CAD Rate: {current_rate:.4f} ({current_date})")
        print(f"Time Horizon: {time_horizon} month(s)")
        
        print(f"\nMethod 1: Original Formula...")
        method1 = self.calculate_original_buffer(df, scenario, volatility_percentile, time_horizon)
        
        print(f"\nMethod 2: EGARCH + Monte Carlo...")
        method2 = self.calculate_egarch_buffer(df, scenario, volatility_percentile, time_horizon)
        
        print(f"\nMethod 3: Enhanced Historical Averages (67/33)...")
        method3 = self.calculate_historical_buffer(df, scenario, volatility_percentile, time_horizon)
        
        print(f"\nMethod 4: Value at Risk...")
        method4 = self.calculate_var_buffer(df, scenario, volatility_percentile, time_horizon)
        
        print(f"\nMethod 5: ARIMA Forecasting...")
        method5 = self.calculate_arima_buffer(df, scenario, volatility_percentile, time_horizon)
        
        methods = [method1, method2, method3, method4, method5]
        valid_methods = [m for m in methods if m is not None]
        
        if not valid_methods:
            return {"error": "All calculation methods failed"}
        
        buffer_prices = [m['buffer_price'] for m in valid_methods]
        buffer_percentages = [m['buffer_percentage'] for m in valid_methods]
        final_rates = [m['final_rate'] for m in valid_methods]
        
        ensemble_stats = {
            'avg_buffer': np.mean(buffer_prices),
            'median_buffer': np.median(buffer_prices),
            'min_buffer': np.min(buffer_prices),
            'max_buffer': np.max(buffer_prices),
            'std_buffer': np.std(buffer_prices),
            'avg_percentage': np.mean(buffer_percentages),
            'avg_final_rate': np.mean(final_rates)
        }
        
        return {
            'current_rate': current_rate,
            'current_date': current_date,
            'scenario': scenario,
            'volatility_percentile': volatility_percentile,
            'time_horizon': time_horizon,
            'methods': valid_methods,
            'ensemble_stats': ensemble_stats,
            'successful_methods': len(valid_methods),
            'df': df
        }

    def create_charts(self, results, save_charts=False, charts_dir="cad_buffer_charts"):
        """Create comprehensive charts for CAD buffer analysis"""
        if not PLOTTING_AVAILABLE:
            print("Matplotlib not available - cannot create charts")
            return False
        
        try:
            if save_charts:
                os.makedirs(charts_dir, exist_ok=True)
                print(f"Charts will be saved to: {charts_dir}/")
            
            df = results['df']
            current_rate = results['current_rate']
            scenario = results['scenario']
            time_horizon = results['time_horizon']
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            fig.suptitle(f'CAD Buffer Analysis - {scenario.upper()} Scenario ({time_horizon}M Horizon)', 
                        fontsize=16, fontweight='bold')
            
            # 1. Historical Rate Chart with Buffer Zones
            ax1 = plt.subplot(3, 3, 1)
            recent_data = df.tail(252)  # Last year of data
            ax1.plot(recent_data['date'], recent_data['rate'], 'b-', linewidth=2, label='USD/CAD Rate')
            ax1.axhline(y=current_rate, color='red', linestyle='--', alpha=0.8, label=f'Current Rate: {current_rate:.4f}')
            
            # Add buffer zones for each method
            colors = ['orange', 'green', 'purple', 'brown', 'pink']
            for i, method in enumerate(results['methods']):
                if method:
                    final_rate = method['final_rate']
                    method_name = method['method'].split(' ')[0]  # Short name
                    ax1.axhline(y=final_rate, color=colors[i], linestyle=':', alpha=0.7, 
                              label=f'{method_name}: {final_rate:.4f}')
            
            ax1.set_title('Historical Rates & Buffer Zones', fontsize=12, fontweight='bold')
            ax1.set_ylabel('USD/CAD Rate')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. Buffer Price Comparison
            ax2 = plt.subplot(3, 3, 2)
            method_names = [m['method'].split(' ')[0] for m in results['methods'] if m]
            buffer_prices = [m['buffer_price'] for m in results['methods'] if m]
            bars = ax2.bar(method_names, buffer_prices, color=colors[:len(buffer_prices)], alpha=0.7)
            ax2.set_title('Buffer Price by Method', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Buffer Price ($)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, price in zip(bars, buffer_prices):
                height = bar.get_height()
                ax2.annotate(f'${price:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
            
            # 3. Buffer Percentage Comparison
            ax3 = plt.subplot(3, 3, 3)
            buffer_percentages = [m['buffer_percentage'] for m in results['methods'] if m]
            bars = ax3.bar(method_names, buffer_percentages, color=colors[:len(buffer_percentages)], alpha=0.7)
            ax3.set_title('Buffer Percentage by Method', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Buffer Percentage (%)')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, pct in zip(bars, buffer_percentages):
                height = bar.get_height()
                ax3.annotate(f'{pct:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
            
            # 4. Rate Volatility Analysis
            ax4 = plt.subplot(3, 3, 4)
            df_vol = df.copy()
            df_vol['daily_return'] = df_vol['rate'].pct_change()
            df_vol['rolling_vol'] = df_vol['daily_return'].rolling(30).std() * np.sqrt(252)
            recent_vol = df_vol.tail(252)
            
            ax4.plot(recent_vol['date'], recent_vol['rolling_vol'] * 100, 'purple', linewidth=2)
            ax4.set_title('30-Day Rolling Volatility (Annualized)', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Volatility (%)')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            # 5. Distribution of Daily Returns
            ax5 = plt.subplot(3, 3, 5)
            daily_returns = df['rate'].pct_change().dropna() * 100
            ax5.hist(daily_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax5.axvline(daily_returns.mean(), color='red', linestyle='--', label=f'Mean: {daily_returns.mean():.3f}%')
            ax5.axvline(daily_returns.mean() + daily_returns.std(), color='orange', linestyle=':', label=f'+1σ: {daily_returns.mean() + daily_returns.std():.3f}%')
            ax5.axvline(daily_returns.mean() - daily_returns.std(), color='orange', linestyle=':', label=f'-1σ: {daily_returns.mean() - daily_returns.std():.3f}%')
            ax5.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Daily Return (%)')
            ax5.set_ylabel('Frequency')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            
            # 6. Ensemble Statistics
            ax6 = plt.subplot(3, 3, 6)
            stats = results['ensemble_stats']
            stat_names = ['Avg', 'Median', 'Min', 'Max']
            stat_values = [stats['avg_buffer'], stats['median_buffer'], stats['min_buffer'], stats['max_buffer']]
            bars = ax6.bar(stat_names, stat_values, color=['blue', 'green', 'red', 'orange'], alpha=0.7)
            ax6.set_title('Ensemble Buffer Statistics', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Buffer Price ($)')
            
            for bar, value in zip(bars, stat_values):
                height = bar.get_height()
                ax6.annotate(f'${value:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
            
            # 7. Method Performance Radar Chart
            ax7 = plt.subplot(3, 3, 7, projection='polar')
            
            # Normalize buffer prices for radar chart (0-1 scale)
            max_buffer = max(buffer_prices)
            normalized_buffers = [bp/max_buffer for bp in buffer_prices]
            
            # Add first point at the end to close the radar chart
            normalized_buffers.append(normalized_buffers[0])
            method_names_radar = method_names + [method_names[0]]
            
            angles = np.linspace(0, 2*np.pi, len(method_names), endpoint=False).tolist()
            angles.append(angles[0])
            
            ax7.plot(angles, normalized_buffers, 'o-', linewidth=2, color='red', alpha=0.7)
            ax7.fill(angles, normalized_buffers, alpha=0.25, color='red')
            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels(method_names, fontsize=10)
            ax7.set_title('Method Comparison (Normalized)', fontsize=12, fontweight='bold')
            ax7.set_ylim(0, 1)
            
            # 8. Rate Trend Analysis
            ax8 = plt.subplot(3, 3, 8)
            
            # Calculate different trend periods
            trend_periods = [30, 60, 90, 180]
            trend_values = []
            
            for period in trend_periods:
                if len(df) >= period:
                    recent_rates = df.tail(period)['rate']
                    trend = (recent_rates.iloc[-1] - recent_rates.iloc[0]) / period * 1000  # Per 1000 days
                    trend_values.append(trend)
                else:
                    trend_values.append(0)
            
            bars = ax8.bar([f'{p}D' for p in trend_periods], trend_values, 
                          color=['lightcoral' if tv < 0 else 'lightgreen' for tv in trend_values], alpha=0.7)
            ax8.set_title('Rate Trends (Different Periods)', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Trend (per 1000 days)')
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax8.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, trend_values):
                height = bar.get_height()
                ax8.annotate(f'{value:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15), textcoords="offset points", 
                           ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
            
            # 9. Summary Statistics Table
            ax9 = plt.subplot(3, 3, 9)
            ax9.axis('off')
            
            # Create summary table data
            table_data = []
            table_data.append(['Metric', 'Value'])
            table_data.append(['Current Rate', f'{current_rate:.4f}'])
            table_data.append(['Scenario', scenario.upper()])
            table_data.append(['Time Horizon', f'{time_horizon} months'])
            table_data.append(['Successful Methods', f'{results["successful_methods"]}/5'])
            table_data.append(['Avg Buffer', f'${stats["avg_buffer"]:.4f}'])
            table_data.append(['Buffer Range', f'${stats["min_buffer"]:.4f} - ${stats["max_buffer"]:.4f}'])
            table_data.append(['Std Deviation', f'${stats["std_buffer"]:.4f}'])
            table_data.append(['Coeff of Variation', f'{(stats["std_buffer"]/stats["avg_buffer"]*100):.1f}%'])
            table_data.append(['67/33 Weighting', 'Applied to Historical'])
            
            table = ax9.table(cellText=table_data[1:], colLabels=table_data[0], 
                             cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            # Style the table
            for i in range(len(table_data)):
                for j in range(len(table_data[0])):
                    cell = table[(i, j)]
                    if i == 0:  # Header row
                        cell.set_facecolor('#4CAF50')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
            
            ax9.set_title('Summary Statistics', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            if save_charts:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{charts_dir}/CAD_Buffer_Analysis_{scenario}_{time_horizon}M_67-33_{timestamp}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"Chart saved as: {filename}")
            
            plt.show()
            
            return True
            
        except Exception as e:
            print(f"Error creating charts: {e}")
            import traceback
            traceback.print_exc()
            return False

    def ask_for_charts(self):
        """Ask user if they want charts and saving options"""
        if not PLOTTING_AVAILABLE:
            print("\nCharts are not available (matplotlib not installed)")
            return False, False, ""
        
        print(f"\n" + "="*60)
        print("CHART OPTIONS")
        print("="*60)
        
        # Ask if charts are wanted
        while True:
            try:
                chart_choice = input("Do you want to generate charts? (y/n): ").strip().lower()
                if chart_choice in ['y', 'yes']:
                    generate_charts = True
                    break
                elif chart_choice in ['n', 'no']:
                    return False, False, ""
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                return False, False, ""
        
        # Ask about saving charts
        while True:
            try:
                save_choice = input("Do you want to save charts to files? (y/n): ").strip().lower()
                if save_choice in ['y', 'yes']:
                    save_charts = True
                    break
                elif save_choice in ['n', 'no']:
                    save_charts = False
                    break
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                return False, False, ""
        
        charts_dir = "cad_buffer_charts"
        if save_charts:
            # Ask for custom directory
            while True:
                try:
                    dir_choice = input(f"Directory for charts (default: {charts_dir}): ").strip()
                    if dir_choice:
                        charts_dir = dir_choice
                    break
                except KeyboardInterrupt:
                    return False, False, ""
        
        return generate_charts, save_charts, charts_dir

    def print_results(self, results):
        if 'error' in results:
            print(f"\nError: {results['error']}")
            return
        
        print(f"\n" + "="*80)
        print(f"CAD BUFFER PRICE CALCULATOR RESULTS (67/33 DIRECTIONAL)")
        print(f"="*80)
        print(f"Date: {results['current_date']}")
        print(f"Current USD/CAD Rate: {results['current_rate']:.4f}")
        print(f"Scenario: {results['scenario'].upper()}")
        print(f"Volatility Percentile: {results['volatility_percentile']}th")
        print(f"Time Horizon: {results['time_horizon']} month(s)")
        print(f"Successful Methods: {results['successful_methods']}/5")
        
        print(f"\n" + "-"*80)
        print(f"METHOD DETAILS")
        print(f"-"*80)
        
        for i, method in enumerate(results['methods'], 1):
            print(f"\n{i}. {method['method']}")
            print(f"   Buffer Price: ${method['buffer_price']:.4f}")
            print(f"   Buffer Percentage: {method['buffer_percentage']:.2f}%")
            print(f"   Final Rate: {method['final_rate']:.4f}")
            print(f"   Data Used: {method['data_periods_used']}")
            
            if 'p1' in method and 'p2' in method:
                print(f"   P1: {method['p1']:.3f}, P2: {method['p2']:.3f}, Vol: {method['vol']:.4f}")
            
            if 'egarch_fitted' in method:
                status = "Fitted" if method['egarch_fitted'] else "Failed (Enhanced Vol Used)"
                print(f"   EGARCH Status: {status}")
            
            # Special display for Historical Averages method with 67/33 weighting
            if 'base_weight' in method and 'directional_weight' in method:
                print(f"   Directional Analysis: {method['directional_type']}")
                print(f"   Up days: {method['up_days']}, Down days: {method['down_days']}")
                print(f"   Weighting: {method['base_weight']:.0%} base + {method['directional_weight']:.0%} directional")
                print(f"   Directional vol: {method['directional_vol']:.4f}, Final vol: {method['weighted_vol']:.4f}")
            
            if 'interpretation' in method:
                print(f"   VaR Profile: {method['interpretation']}")
            
            if 'best_order' in method:
                print(f"   ARIMA Order: {method['best_order']}")
                if 'scenario_agreement' in method:
                    print(f"   Forecast Agreement: {method['scenario_agreement']}")
                if 'short_window' in method and 'medium_window' in method:
                    print(f"   Windows: Short={method['short_window']}m, Medium={method['medium_window']}m")
        
        print(f"\n" + "-"*80)
        print(f"ENSEMBLE STATISTICS")
        print(f"-"*80)
        stats = results['ensemble_stats']
        print(f"Average Buffer Price: ${stats['avg_buffer']:.4f}")
        print(f"Median Buffer Price: ${stats['median_buffer']:.4f}")
        print(f"Min Buffer Price: ${stats['min_buffer']:.4f}")
        print(f"Max Buffer Price: ${stats['max_buffer']:.4f}")
        print(f"Standard Deviation: ${stats['std_buffer']:.4f}")
        print(f"Average Buffer %: {stats['avg_percentage']:.2f}%")
        print(f"Average Final Rate: {stats['avg_final_rate']:.4f}")
        print(f"Coefficient of Variation: {(stats['std_buffer']/stats['avg_buffer']*100):.1f}%")
        
        print(f"\n" + "-"*80)
        print(f"DIRECTIONAL WEIGHTING SUMMARY")
        print(f"-"*80)
        print(f"Enhanced Historical Averages now uses 67% base volatility + 33% directional volatility")
        print(f"This provides modest directional bias while maintaining statistical robustness")
        print(f"Buffers should now differ slightly between depreciating vs appreciating scenarios")
        
        print(f"\n" + "="*80)

def main():
    print("CAD Buffer Price Calculator - 5 Methods with 67/33 Directional Weighting")
    print("="*80)
    
    calculator = CADBufferCalculator()
    
    # Option to run diagnostic first
    print("\nDIAGNOSTIC OPTIONS:")
    print("1. Run full calculation")
    print("2. Run diagnostic mode first")
    
    while True:
        try:
            diag_choice = input("Select option (1 or 2): ").strip()
            if diag_choice == '1':
                break
            elif diag_choice == '2':
                calculator.diagnose_data_issue()
                print("\nDiagnostic complete. Proceeding to full calculation...\n")
                break
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            return
    
    while True:
        print("\nCONFIGURATION")
        print("-" * 30)
        
        print("1. CAD Depreciating (USD/CAD rate increases)")
        print("2. CAD Appreciating (USD/CAD rate decreases)")
        while True:
            try:
                scenario_choice = input("\nSelect scenario (1 or 2): ").strip()
                if scenario_choice == '1':
                    scenario = 'depreciating'
                    break
                elif scenario_choice == '2':
                    scenario = 'appreciating'
                    break
                else:
                    print("Please enter 1 or 2")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return
        
        print("\nVolatility Percentile Options:")
        print("25 - Conservative (25th percentile)")
        print("50 - Moderate (50th percentile)")
        print("75 - Aggressive (75th percentile)")
        print("90 - Very Aggressive (90th percentile)")
        
        while True:
            try:
                vol_input = input("Select volatility percentile (25/50/75/90): ").strip()
                if vol_input in ['25', '50', '75', '90']:
                    volatility_percentile = int(vol_input)
                    break
                else:
                    print("Please enter 25, 50, 75, or 90")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return
        
        print("\nTime Horizon Options:")
        print("1 - 1 Month")
        print("3 - 3 Months")
        print("6 - 6 Months")
        print("12 - 12 Months")
        
        while True:
            try:
                time_input = input("Select time horizon (1/3/6/12): ").strip()
                if time_input in ['1', '3', '6', '12']:
                    time_horizon = int(time_input)
                    break
                else:
                    print("Please enter 1, 3, 6, or 12")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return
        
        print(f"\nCALCULATING BUFFERS WITH 67/33 DIRECTIONAL WEIGHTING...")
        print(f"Scenario: {scenario.upper()}")
        print(f"Volatility: {volatility_percentile}th percentile")
        print(f"Time Horizon: {time_horizon} month(s)")
        
        try:
            results = calculator.calculate_all_methods(scenario, volatility_percentile, time_horizon)
            calculator.print_results(results)
            
            # Ask about charts
            if 'error' not in results:
                generate_charts, save_charts, charts_dir = calculator.ask_for_charts()
                
                if generate_charts:
                    print(f"\nGenerating charts...")
                    success = calculator.create_charts(results, save_charts, charts_dir)
                    if success:
                        print("Charts generated successfully!")
                    else:
                        print("Chart generation failed.")
            
        except Exception as e:
            print(f"\nCalculation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n" + "="*80)
        while True:
            try:
                continue_choice = input("Run another calculation? (y/n): ").strip().lower()
                if continue_choice in ['y', 'yes']:
                    print("\n" + "="*80)
                    break
                elif continue_choice in ['n', 'no']:
                    print("\nThank you for using CAD Buffer Price Calculator!")
                    return
                else:
                    print("Please enter 'y' or 'n'")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return

if __name__ == "__main__":
    main()