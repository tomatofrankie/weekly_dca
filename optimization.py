import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

today = datetime.today().strftime('%Y-%m-%d')

# -----------------------------
# 1. Download Data (same as backtest.py)
# -----------------------------
print("Fetching market data...")
period = "1y"
spy = yf.download("SPY", period=period, end=today, progress=False)['Close']
vix = yf.download("^VIX", period=period, end=today, progress=False)['Close']

spy = spy.dropna()
vix = vix.dropna()

data = pd.concat([spy, vix], axis=1).dropna()
data.columns = ['SPY', 'VIX']
print(f"Data fetched: {len(data)} days")

# -----------------------------
# 2. Backtest Function (from backtest.py)
# -----------------------------
def backtest_retracement(data, freq="W", base_contrib=1000, 
                         vix_threshold=20, retracement_levels=[0.10, 0.20, 0.30]):
    prices = data["SPY"].resample(freq).last()
    vix_vals = data["VIX"].resample(freq).last()
    
    shares, cash = 0.0, 0.0
    history = []
    rolling_high = prices.expanding().max()

    for date, price in prices.items():
        vix_now = vix_vals.loc[date]
        high = rolling_high.loc[date]
        retracement = (high - price) / high if high > 0 else 0
        
        mult = 1
        if vix_now > vix_threshold:
            if retracement >= retracement_levels[2]:
                mult = 8
            elif retracement >= retracement_levels[1]:
                mult = 4
            elif retracement >= retracement_levels[0]:
                mult = 2
        
        invest = base_contrib * mult
        buy_shares = invest / price
        shares += buy_shares
        cash -= invest

        history.append((date, price, shares, cash, shares*price + cash, 
                       invest, mult, vix_now, retracement*100, high))
    
    results = pd.DataFrame(history, 
                           columns=["Date","Price","Shares","Cash",
                                    "PortfolioValue","Contribution","Multiplier",
                                    "VIX","Retracement%","RollingHigh"])
    results.set_index("Date", inplace=True)
    return results

# -----------------------------
# 3. Performance Metrics Function (with error handling)
# -----------------------------
def calculate_metrics(results):
    """Calculate performance metrics with proper NaN/inf handling"""
    try:
        final_value = results["PortfolioValue"].iloc[-1]
        total_invested = results["Contribution"].sum()
        
        # Basic checks
        if pd.isna(final_value) or pd.isna(total_invested) or total_invested == 0:
            return None
        
        years = (results.index[-1] - results.index[0]).days / 365.25
        if years == 0:
            return None
            
        cagr = (final_value / total_invested) ** (1/years) - 1 if total_invested > 0 else 0
        
        # Drawdown calculation with safety checks
        portfolio_values = results["PortfolioValue"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(portfolio_values) == 0:
            return None
            
        cummax = portfolio_values.cummax()
        drawdown = (cummax - portfolio_values) / cummax
        drawdown = drawdown.replace([np.inf, -np.inf], np.nan).dropna()
        max_dd = drawdown.max() if len(drawdown) > 0 else 0
        
        # Returns calculation with safety checks
        results_copy = results.copy()
        results_copy['Returns'] = results_copy['PortfolioValue'].pct_change()
        results_copy['Returns'] = results_copy['Returns'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(results_copy['Returns']) < 2:
            sharpe = 0
            sortino = 0
        else:
            returns_mean = results_copy['Returns'].mean()
            returns_std = results_copy['Returns'].std()
            
            # Sharpe Ratio
            if pd.notna(returns_std) and returns_std > 0:
                sharpe = (returns_mean / returns_std) * np.sqrt(52)
            else:
                sharpe = 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = results_copy['Returns'][results_copy['Returns'] < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std()
                if pd.notna(downside_std) and downside_std > 0:
                    sortino = (returns_mean / downside_std) * np.sqrt(52)
                else:
                    sortino = 0
            else:
                sortino = sharpe  # If no downside, use Sharpe
        
        # Calmar Ratio
        if pd.notna(max_dd) and abs(max_dd) > 0.001:
            calmar = cagr / abs(max_dd)
        else:
            calmar = 0
        
        # Final validation
        if any(pd.isna(x) or np.isinf(x) for x in [cagr, max_dd, sharpe, sortino, calmar]):
            return None
        
        return {
            'final_value': final_value,
            'total_invested': total_invested,
            'net_profit': final_value - total_invested,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'years': years
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

# -----------------------------
# 4. Optimization Grid Search
# -----------------------------
def optimize_parameters(data, freq="W", base_contrib=1000):
    """
    Test multiple parameter combinations and return results
    """
    # Define parameter ranges to test
    vix_thresholds = [15, 18, 20, 22, 25, 28, 30]
    retracement_10_levels = [0.08]#[0.05, 0.08, 0.10, 0.12, 0.15]
    retracement_20_levels = [0.16]#[0.1, 0.16, 0.20, 0.24, 0.3]
    retracement_30_levels = [0.24]#[0.15, 0.24, 0.30, 0.36, 0.45]
    
    results_list = []
    
    # Calculate total valid combinations
    valid_count = 0
    for ret_10 in retracement_10_levels:
        for ret_20 in retracement_20_levels:
            for ret_30 in retracement_30_levels:
                if ret_10 < ret_20 < ret_30:
                    valid_count += 1
    
    total_combinations = len(vix_thresholds) * valid_count
    
    print(f"\nTesting {total_combinations} parameter combinations...")
    print("="*60)
    
    counter = 0
    successful = 0
    
    for vix_thresh in vix_thresholds:
        for ret_10 in retracement_10_levels:
            for ret_20 in retracement_20_levels:
                for ret_30 in retracement_30_levels:
                    # Skip invalid combinations where levels aren't increasing
                    if ret_10 >= ret_20 or ret_20 >= ret_30:
                        continue
                    
                    counter += 1
                    if counter % 50 == 0:
                        print(f"Progress: {counter}/{total_combinations} combinations tested... ({successful} successful)")
                    
                    try:
                        retracement_levels = [ret_10, ret_20, ret_30]
                        backtest_results = backtest_retracement(
                            data, freq=freq, base_contrib=base_contrib,
                            vix_threshold=vix_thresh,
                            retracement_levels=retracement_levels
                        )
                        
                        metrics = calculate_metrics(backtest_results)
                        
                        if metrics is None:
                            continue
                        
                        results_list.append({
                            'vix_threshold': vix_thresh,
                            'retracement_10': ret_10,
                            'retracement_20': ret_20,
                            'retracement_30': ret_30,
                            'final_value': metrics['final_value'],
                            'total_invested': metrics['total_invested'],
                            'net_profit': metrics['net_profit'],
                            'cagr': metrics['cagr'],
                            'max_drawdown': metrics['max_drawdown'],
                            'sharpe_ratio': metrics['sharpe_ratio'],
                            'sortino_ratio': metrics['sortino_ratio'],
                            'calmar_ratio': metrics['calmar_ratio']
                        })
                        successful += 1
                        
                    except Exception as e:
                        print(f"Error with params: VIX={vix_thresh}, Ret={[ret_10, ret_20, ret_30]}: {e}")
                        continue
    
    print(f"\nCompleted: {successful} successful backtests out of {counter} tested")
    return pd.DataFrame(results_list)

# -----------------------------
# 5. Run Optimization
# -----------------------------
print("\n" + "="*60)
print("PARAMETER OPTIMIZATION FOR SPY MARTINGALE STRATEGY")
print("="*60)

optimization_results = optimize_parameters(data, freq="M", base_contrib=1000)

if len(optimization_results) == 0:
    print("ERROR: No valid optimization results generated!")
    exit(1)

print(f"\nCompleted testing {len(optimization_results)} valid parameter combinations")

# -----------------------------
# 6. Analyze Results
# -----------------------------
# Remove any remaining invalid values
optimization_results = optimization_results.replace([np.inf, -np.inf], np.nan).dropna()

if len(optimization_results) == 0:
    print("ERROR: All optimization results contain invalid values!")
    exit(1)

# Sort by different metrics
top_by_cagr = optimization_results.nlargest(10, 'cagr')
top_by_sharpe = optimization_results.nlargest(10, 'sharpe_ratio')
top_by_calmar = optimization_results.nlargest(10, 'calmar_ratio')
top_by_profit = optimization_results.nlargest(10, 'net_profit')

print("\n" + "="*60)
print("TOP 10 PARAMETER SETS BY CAGR")
print("="*60)
print(top_by_cagr[['vix_threshold', 'retracement_10', 'retracement_20', 
                    'retracement_30', 'cagr', 'max_drawdown', 'sharpe_ratio']].to_string(index=False))

print("\n" + "="*60)
print("TOP 10 PARAMETER SETS BY SHARPE RATIO")
print("="*60)
print(top_by_sharpe[['vix_threshold', 'retracement_10', 'retracement_20', 
                      'retracement_30', 'cagr', 'max_drawdown', 'sharpe_ratio']].to_string(index=False))

print("\n" + "="*60)
print("TOP 10 PARAMETER SETS BY CALMAR RATIO")
print("="*60)
print(top_by_calmar[['vix_threshold', 'retracement_10', 'retracement_20', 
                      'retracement_30', 'cagr', 'max_drawdown', 'calmar_ratio']].to_string(index=False))

# Find the best overall (balanced approach)
optimization_results['composite_score'] = (
    optimization_results['cagr'] * 0.4 + 
    optimization_results['sharpe_ratio'] * 0.3 + 
    optimization_results['calmar_ratio'] * 0.3
)
best_overall = optimization_results.nlargest(1, 'composite_score').iloc[0]

print("\n" + "="*60)
print("RECOMMENDED OPTIMAL PARAMETERS (Composite Score)")
print("="*60)
print(f"VIX Threshold: {best_overall['vix_threshold']}")
print(f"Retracement Level 1 (2x): {best_overall['retracement_10']:.2%}")
print(f"Retracement Level 2 (4x): {best_overall['retracement_20']:.2%}")
print(f"Retracement Level 3 (8x): {best_overall['retracement_30']:.2%}")
print(f"\nPerformance Metrics:")
print(f"  CAGR: {best_overall['cagr']:.2%}")
print(f"  Max Drawdown: {best_overall['max_drawdown']:.2%}")
print(f"  Sharpe Ratio: {best_overall['sharpe_ratio']:.2f}")
print(f"  Calmar Ratio: {best_overall['calmar_ratio']:.2f}")
print(f"  Final Value: ${best_overall['final_value']:,.2f}")
print(f"  Net Profit: ${best_overall['net_profit']:,.2f}")

# -----------------------------
# 7. Visualization of Optimization Results
# -----------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Parameter Optimization Analysis', fontsize=16, fontweight='bold')

# Plot 1: CAGR vs Max Drawdown (scatter)
scatter1 = axes[0, 0].scatter(optimization_results['max_drawdown'] * 100, 
                              optimization_results['cagr'] * 100,
                              c=optimization_results['sharpe_ratio'], 
                              cmap='RdYlGn', s=50, alpha=0.6)
axes[0, 0].scatter(best_overall['max_drawdown'] * 100, best_overall['cagr'] * 100,
                   color='red', s=200, marker='*', edgecolors='black', linewidths=2,
                   label='Optimal', zorder=5)
axes[0, 0].set_xlabel('Max Drawdown (%)')
axes[0, 0].set_ylabel('CAGR (%)')
axes[0, 0].set_title('CAGR vs Max Drawdown (colored by Sharpe Ratio)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0, 0], label='Sharpe Ratio')

# Plot 2: VIX Threshold impact on CAGR
vix_grouped = optimization_results.groupby('vix_threshold')['cagr'].agg(['mean', 'std', 'max'])
axes[0, 1].errorbar(vix_grouped.index, vix_grouped['mean'] * 100, 
                    yerr=vix_grouped['std'] * 100, fmt='o-', capsize=5, capthick=2)
axes[0, 1].scatter(vix_grouped.index, vix_grouped['max'] * 100, 
                   color='green', marker='^', s=100, alpha=0.6, label='Max CAGR')
axes[0, 1].set_xlabel('VIX Threshold')
axes[0, 1].set_ylabel('CAGR (%)')
axes[0, 1].set_title('Impact of VIX Threshold on CAGR')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Heatmap of CAGR by retracement levels (fixing VIX at optimal)
optimal_vix = best_overall['vix_threshold']
filtered_results = optimization_results[optimization_results['vix_threshold'] == optimal_vix]
if len(filtered_results) > 0:
    pivot_cagr = filtered_results.pivot_table(values='cagr', 
                                              index='retracement_20', 
                                              columns='retracement_10', 
                                              aggfunc='mean')
    sns.heatmap(pivot_cagr * 100, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=axes[1, 0], cbar_kws={'label': 'CAGR (%)'})
    axes[1, 0].set_title(f'CAGR Heatmap (VIX={optimal_vix})')
    axes[1, 0].set_xlabel('10% Retracement Level')
    axes[1, 0].set_ylabel('20% Retracement Level')
else:
    axes[1, 0].text(0.5, 0.5, 'Insufficient data for heatmap', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)

# Plot 4: Composite score distribution
axes[1, 1].hist(optimization_results['composite_score'], bins=50, 
                color='steelblue', edgecolor='black', alpha=0.7)
axes[1, 1].axvline(best_overall['composite_score'], color='red', 
                   linestyle='--', linewidth=2, label='Optimal')
axes[1, 1].set_xlabel('Composite Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Distribution of Composite Scores')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_results.png', dpi=300, bbox_inches='tight')
print("\n" + "="*60)
print("Optimization chart saved as 'optimization_results.png'")
print("="*60)
plt.show()

# -----------------------------
# 8. Save detailed results to CSV
# -----------------------------
optimization_results.to_csv('optimization_results.csv', index=False)
print("Detailed results saved to 'optimization_results.csv'")

# -----------------------------
# 9. Test optimal parameters
# -----------------------------
print("\n" + "="*60)
print("RUNNING BACKTEST WITH OPTIMAL PARAMETERS")
print("="*60)

optimal_backtest = backtest_retracement(
    data, freq="W", base_contrib=1000,
    vix_threshold=best_overall['vix_threshold'],
    retracement_levels=[best_overall['retracement_10'], 
                       best_overall['retracement_20'],
                       best_overall['retracement_30']]
)

optimal_metrics = calculate_metrics(optimal_backtest)

if optimal_metrics:
    print("\nOptimal Strategy Performance:")
    for key, value in optimal_metrics.items():
        if isinstance(value, float):
            if 'ratio' in key or 'cagr' in key or 'drawdown' in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"  {key.replace('_', ' ').title()}: ${value:,.2f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}")
else:
    print("ERROR: Could not calculate metrics for optimal parameters")

print("\n" + "="*60)
print("Optimization complete!")
print("="*60)