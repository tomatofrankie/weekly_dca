import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 

today = datetime.today().strftime('%Y-%m-%d')

# -----------------------------
# 1. Download Data
# -----------------------------
period = "50y"
spy = yf.download("^SPX", period=period, end=today, progress=False)['Close']
vix = yf.download("^VIX", period=period, end=today, progress=False)['Close']

print(f"Data fetched: {len(spy)} SPY days, {len(vix)} VIX days")

# Ensure both are Series with DateTimeIndex 
spy = spy.dropna()
vix = vix.dropna()

# Merge into DataFrame (align by index)
data = pd.concat([spy, vix], axis=1).dropna()
data.columns = ['SPY', 'VIX']

# -----------------------------
# 2. Backtest Function with Retracement Logic
# -----------------------------
def backtest_retracement(data, freq="W", base_contrib=1000, 
                         vix_threshold=18, retracement_levels=[0.08, 0.16, 0.24]):
    """
    Martingale triggers based on:
    - VIX > threshold AND
    - SPY retracement from recent high
    
    Multipliers:
    - Base: 1x (VIX <= threshold or no significant retracement)
    - 2x: VIX > 20 and SPY down 10% from high
    - 4x: VIX > 20 and SPY down 20% from high
    - 8x: VIX > 20 and SPY down 30%+ from high
    """
    # Resample to the correct frequency (end of period prices)
    prices = data["SPY"].resample(freq).last()
    vix_vals = data["VIX"].resample(freq).last()
    
    shares, cash = 0.0, 0.0
    history = []
    
    # Calculate rolling high for retracement calculation
    rolling_high = prices.expanding().max()

    for date, price in prices.items():
        vix_now = vix_vals.loc[date]
        high = rolling_high.loc[date]
        
        # Calculate retracement percentage from high
        retracement = (high - price) / high if high > 0 else 0
        
        # Determine multiplier based on VIX and retracement
        mult = 1  # Default
        
        if vix_now > vix_threshold:
            # VIX is elevated, check retracement levels
            if retracement >= retracement_levels[2]:  # 30%+ drawdown
                mult = 8
            elif retracement >= retracement_levels[1]:  # 20%+ drawdown
                mult = 4
            elif retracement >= retracement_levels[0]:  # 10%+ drawdown
                mult = 2
            # else: mult stays 1 (VIX high but no significant retracement)
        
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
# 3. Run Both Strategies
# -----------------------------
print("\n" + "="*60)
print("Running Backtests with Retracement-Based Martingale")
print("="*60)

res_weekly = backtest_retracement(data, freq="W", base_contrib=250)
res_monthly = backtest_retracement(data, freq="ME", base_contrib=1000)#, vix_threshold=18, retracement_levels=[0.08, 0.16, 0.24])

# -----------------------------
# 4. Performance Metrics
# -----------------------------
def performance_stats(results):
    final_value = results["PortfolioValue"].iloc[-1]
    total_invested = results["Contribution"].sum()
    
    # Calculate years
    years = (results.index[-1] - results.index[0]).days / 365.25
    cagr = (final_value / total_invested) ** (1/years) - 1 if years > 0 else 0

    # Drawdown calculation
    cummax = results["PortfolioValue"].cummax().replace(0, np.nan)
    drawdown = (cummax - results["PortfolioValue"]) / cummax
    max_dd = drawdown.max()
    
    # Count trades by multiplier
    mult_counts = results["Multiplier"].value_counts().sort_index()
    
    return {
        "Final Value": round(final_value, 2),
        "Total Invested": round(total_invested, 2),
        "Net Profit": round(final_value - total_invested, 2),
        "CAGR %": round(cagr*100, 2),
        "Max Drawdown %": round(max_dd*100, 2),
        "Years": round(years, 2),
        "Total Contributions": len(results),
        "Multiplier Distribution": mult_counts.to_dict()
    }

print("\n" + "="*60)
print("WEEKLY STRATEGY RESULTS")
print("="*60)
weekly_stats = performance_stats(res_weekly)
for key, value in weekly_stats.items():
    print(f"{key}: {value}")

print("\n" + "="*60)
print("MONTHLY STRATEGY RESULTS")
print("="*60)
monthly_stats = performance_stats(res_monthly)
for key, value in monthly_stats.items():
    print(f"{key}: {value}")

# -----------------------------
# 5. Detailed Visualizations
# -----------------------------
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('SPY DCA with VIX + Retracement-Based Martingale', fontsize=16, fontweight='bold')

# Plot 1: Portfolio Value Comparison
axes[0, 0].plot(res_weekly.index, res_weekly["PortfolioValue"], 
                label="Weekly", linewidth=2, color='blue')
axes[0, 0].plot(res_monthly.index, res_monthly["PortfolioValue"], 
                label="Monthly", linewidth=2, color='green')
axes[0, 0].set_title("Portfolio Value Over Time")
axes[0, 0].set_ylabel("Portfolio Value ($)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: SPY Price with Contributions (Weekly)
ax2 = axes[0, 1]
ax2_twin = ax2.twinx()
ax2.plot(res_weekly.index, res_weekly["Price"], color='black', 
         linewidth=1.5, label='SPY Price')
scatter = ax2_twin.scatter(res_weekly.index, res_weekly["Contribution"], 
                           c=res_weekly["Multiplier"], cmap='YlOrRd', 
                           s=50, alpha=0.6, label='Contribution Size')
ax2.set_title("SPY Price & Weekly Contributions")
ax2.set_ylabel("SPY Price ($)", color='black')
ax2_twin.set_ylabel("Contribution ($)", color='red')
ax2.legend(loc='upper left')
plt.colorbar(scatter, ax=ax2_twin, label='Multiplier')
ax2.grid(True, alpha=0.3)

# Plot 3: VIX Over Time with Threshold
axes[1, 0].plot(res_weekly.index, res_weekly["VIX"], 
                color='orange', linewidth=1.5)
axes[1, 0].axhline(y=20, color='red', linestyle='--', 
                   linewidth=2, label='VIX Threshold (20)')
axes[1, 0].set_title("VIX Level Over Time")
axes[1, 0].set_ylabel("VIX")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Retracement Levels (Weekly)
axes[1, 1].plot(res_weekly.index, res_weekly["Retracement%"], 
                color='purple', linewidth=1.5)
axes[1, 1].axhline(y=8, color='yellow', linestyle='--', 
                   linewidth=1.5, label='8% (2x trigger)')
axes[1, 1].axhline(y=16, color='orange', linestyle='--', 
                   linewidth=1.5, label='16% (4x trigger)')
axes[1, 1].axhline(y=24, color='red', linestyle='--', 
                   linewidth=1.5, label='24% (8x trigger)')
axes[1, 1].set_title("SPY Retracement from High")
axes[1, 1].set_ylabel("Retracement (%)")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 5: Multiplier Distribution (Weekly)
mult_dist_weekly = res_weekly["Multiplier"].value_counts().sort_index()
axes[2, 0].bar(mult_dist_weekly.index, mult_dist_weekly.values, 
               color=['green', 'yellow', 'orange', 'red'][:len(mult_dist_weekly)])
axes[2, 0].set_title("Weekly Multiplier Distribution")
axes[2, 0].set_xlabel("Multiplier")
axes[2, 0].set_ylabel("Number of Contributions")
axes[2, 0].grid(True, alpha=0.3, axis='y')

# Plot 6: Drawdown Analysis (Weekly)
cummax = res_weekly["PortfolioValue"].cummax()
drawdown = (cummax - res_weekly["PortfolioValue"]) / cummax * 100
axes[2, 1].fill_between(res_weekly.index, drawdown, 0, 
                        alpha=0.7, color='red')
axes[2, 1].set_title("Portfolio Drawdown (Weekly)")
axes[2, 1].set_ylabel("Drawdown (%)")
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('SPY_retracement_martingale_backtest.png', dpi=300, bbox_inches='tight')
print("\n" + "="*60)
print("Chart saved as 'SPY_retracement_martingale_backtest.png'")
print("="*60)
plt.show()

# -----------------------------
# 6. Summary Statistics Table
# -----------------------------
print("\n" + "="*60)
print("DETAILED TRADE ANALYSIS (Weekly Strategy)")
print("="*60)
print(f"\nTrades by Multiplier:")
for mult, count in res_weekly["Multiplier"].value_counts().sort_index().items():
    pct = count / len(res_weekly) * 100
    total_contrib = (res_weekly[res_weekly["Multiplier"] == mult]["Contribution"].sum())
    print(f"  {mult}x: {count} trades ({pct:.1f}%) - Total: ${total_contrib:,.0f}")

print(f"\nMax single contribution: ${res_weekly['Contribution'].max():,.0f}")
print(f"Average contribution: ${res_weekly['Contribution'].mean():,.0f}")
print(f"Max retracement observed: {res_weekly['Retracement%'].max():.2f}%")