import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import logging
from pathlib import Path
import tg

# -----------------------------
# Logging Configuration
# -----------------------------
def setup_logging():
    """Configure logging to file and console"""
    # Create logs directory if it doesn't exist
    log = Path("log")
    log.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = log / f"dca_{datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Create logger
    logger = logging.getLogger('dca')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler - logs everything
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Console handler - logs everything
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# -----------------------------
# Configuration
# -----------------------------
VIX_THRESHOLD = 18
RETRACEMENT_LEVELS = [0.08, 0.16, 0.24]  # 8%, 16%, 24%
BASE_CONTRIBUTION = 1000  # Monthly base amount

# Multiplier mapping
MULTIPLIERS = {
    'base': 1,
    'level1': 2,   # 8% retracement
    'level2': 4,   # 16% retracement
    'level3': 8    # 24% retracement
}

# Log configuration on startup
logger.info("=" * 60)
logger.info("Configuration loaded:")
logger.info(f"  VIX Threshold: {VIX_THRESHOLD}")
logger.info(f"  Retracement Levels: {RETRACEMENT_LEVELS}")
#logger.info(f"  Base Contribution: ${BASE_CONTRIBUTION}")
logger.info(f"  Multipli#ers: {MULTIPLIERS}")
logger.info("=" * 60)

# -----------------------------
# Fetch Current Market Data
# -----------------------------
def get_current_market_data():
    """Fetch current VOO price, VIX, and calculate retracement"""
    logger.info("Fetching current market data...")
    logger.info("-" * 60)
    
    try:
        # Get current VIX
        logger.info("Downloading VIX data...")
        vix_data = yf.download("^VIX", period="5d", progress=False)
        current_vix = vix_data['Close'].iloc[-1]['^VIX']
        logger.info(f"VIX data retrieved: {current_vix:.2f}")
        
        # Get SPY data for last year to calculate rolling high
        logger.info("Downloading SPY data (1 year)...")
        spy_data = yf.download("VOO", period="1y", progress=False)
        current_spy = spy_data['Close'].iloc[-1]['VOO']
        rolling_high = spy_data['Close'].max()['VOO']
        logger.info(f"SPY current price: ${current_spy:.2f}")
        logger.info(f"SPY 52-week high: ${rolling_high:.2f}")
        
        # Calculate retracement
        retracement = (rolling_high - current_spy) / rolling_high if rolling_high > 0 else 0
        retracement_pct = retracement * 100
        logger.info(f"Calculated retracement: {retracement_pct:.2f}%")
        
        # Get last update time
        last_date = spy_data.index[-1]
        logger.info(f"Last market update: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        market_data = {
            'current_vix': current_vix,
            'current_spy': current_spy,
            'rolling_high': rolling_high,
            'retracement': retracement,
            'retracement_pct': retracement_pct,
            'last_update': last_date
        }
        
        logger.info("Market data fetch completed successfully")
        return market_data
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}", exc_info=True)
        sys.exit(1)

# -----------------------------
# Calculate Required Contribution
# -----------------------------
def calculate_contribution(market_data, base_contrib=BASE_CONTRIBUTION, 
                          vix_threshold=VIX_THRESHOLD, 
                          retracement_levels=RETRACEMENT_LEVELS):
    """
    Calculate the contribution multiplier based on:
    - VIX > threshold AND
    - SPY retracement from recent high
    """
    logger.info("Calculating contribution multiplier...")
    
    vix = market_data['current_vix']
    retracement = market_data['retracement']
    
    logger.info(f"Input parameters - VIX: {vix:.2f}, Retracement: {retracement*100:.2f}%")
    
    # Default multiplier
    multiplier = 1
    trigger_reason = "Normal market conditions"
    
    # Check if VIX is elevated
    if vix > vix_threshold:
        logger.info(f"VIX ({vix:.2f}) exceeds threshold ({vix_threshold})")
        # VIX is elevated, check retracement levels
        if retracement >= retracement_levels[2]:  # 24%+ drawdown
            multiplier = 8
            trigger_reason = f"VIX > {vix_threshold} AND SPY down {retracement*100:.1f}% from high (≥24%)"
            logger.info(f"Level 3 trigger: Multiplier = 8x (Retracement ≥24%)")
        elif retracement >= retracement_levels[1]:  # 16%+ drawdown
            multiplier = 4
            trigger_reason = f"VIX > {vix_threshold} AND SPY down {retracement*100:.1f}% from high (≥16%)"
            logger.info(f"Level 2 trigger: Multiplier = 4x (Retracement ≥16%)")
        elif retracement >= retracement_levels[0]:  # 8%+ drawdown
            multiplier = 2
            trigger_reason = f"VIX > {vix_threshold} AND SPY down {retracement*100:.1f}% from high (≥8%)"
            logger.info(f"Level 1 trigger: Multiplier = 2x (Retracement ≥8%)")
        else:
            trigger_reason = f"VIX > {vix_threshold} but retracement only {retracement*100:.1f}% (< 8%)"
            logger.info(f"VIX elevated but retracement insufficient: {retracement*100:.1f}%")
    else:
        trigger_reason = f"VIX below threshold ({vix:.2f} ≤ {vix_threshold})"
        logger.info(f"VIX below threshold: {vix:.2f} ≤ {vix_threshold}")
    
    contribution_amount = base_contrib * multiplier
    # logger.info(f"Final calculation: ${base_contrib} × {multiplier} = ${contribution_amount}")
    
    contribution_info = {
        'multiplier': multiplier,
        'contribution_amount': contribution_amount,
        'trigger_reason': trigger_reason
    }
    
    logger.info("Contribution calculation completed")
    return contribution_info

# -----------------------------
# Display Notification
# -----------------------------
def display_notification(market_data, contribution_info, new_entry):
    """Display formatted notification with current market status"""
    
    logger.info("\n" + "=" * 60)
    logger.info("📊 SPY MARTINGALE DCA - CURRENT CONTRIBUTION ALERT")
    logger.info("=" * 60)
    
    logger.info(f"\n📅 Last Market Update: {market_data['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info("\n📈 CURRENT MARKET CONDITIONS:")
    logger.info("-" * 60)
    logger.info(f"  VOO Price:           ${market_data['current_spy']:.2f}")
    logger.info(f"  52-Week High:        ${market_data['rolling_high']:.2f}")
    logger.info(f"  Retracement:         {market_data['retracement_pct']:.2f}%")
    logger.info(f"  VIX Level:           {market_data['current_vix']:.2f}")
    #logger.info(f"  VIX Threshold:       {VIX_THRESHOLD}")
    
    logger.info("\n💰 CONTRIBUTION RECOMMENDATION:")
    logger.info("-" * 60)
    #logger.info(f"  Base Contribution:   ${BASE_CONTRIBUTION:,.0f}")
    logger.info(f"  Multiplier:          {contribution_info['multiplier']}x")
    #logger.info(f"  TOTAL TO INVEST:     ${contribution_info['contribution_amount']:,.0f}")

    logger.info("\n💵 Current Portfolio:")
    logger.info(f"  Entry Price:         ${new_entry['Entry Price']:.2f}")
    logger.info(f"  Total Contribution:  ${new_entry['Contribution']}")
    logger.info(f"  Portfolio Value:     ${new_entry['Portfolio Value']}")
    
    # Send telegram notification
    try:
        asyncio.run(tg.send_to_channel(contribution_info['multiplier'], new_entry))
        logger.info("Telegram notification sent successfully")
    except Exception as e:
        logger.error(f"Failed to send Telegram notification: {e}", exc_info=True)
    
    logger.info("\n🎯 TRIGGER ANALYSIS:")
    logger.info("-" * 60)
    logger.info(f"  {contribution_info['trigger_reason']}")
    
    # Visual indicator
    logger.info("\n📊 RETRACEMENT LEVELS:")
    logger.info("-" * 60)
    current_ret = market_data['retracement_pct']
    
    levels = [
        (0, 8, "1x - Normal", "🟢"),
        (8, 16, "2x - Moderate Dip", "🟡"),
        (16, 24, "4x - Significant Dip", "🟠"),
        (24, 100, "8x - Major Dip", "🔴")
    ]
    
    for low, high, desc, emoji in levels:
        is_current = low <= current_ret < high if high < 100 else current_ret >= low
        marker = " ← YOU ARE HERE" if is_current else ""
        logger.info(f"  {emoji} {low}%-{high if high < 100 else '∞'}%: {desc}{marker}")
    
    logger.info("\n" + "=" * 60)
    
    # Alert emphasis
    if contribution_info['multiplier'] > 1:
        logger.warning("⚠️  ELEVATED MULTIPLIER ACTIVE - BUYING OPPORTUNITY! ⚠️")
    else:
        logger.info("✅ Normal contribution period - standard DCA amount")
    
    logger.info("=" * 60 + "\n")

# -----------------------------
# Additional Analysis
# -----------------------------
def show_historical_context(market_data):
    """Show how current conditions compare to historical data"""
    logger.info("\n📊 HISTORICAL CONTEXT (Past 5 Years):")
    logger.info("-" * 60)
    
    try:
        # Get 5 year data
        logger.info("Fetching 5-year historical data...")
        spy_5y = yf.download("VOO", period="5y", progress=False)['Close']
        vix_5y = yf.download("^VIX", period="5y", progress=False)['Close']
        
        # Calculate percentiles
        current_spy = market_data['current_spy']
        current_vix = market_data['current_vix']
        
        spy_percentile = (spy_5y['VOO'] < current_spy).sum() / len(spy_5y) * 100
        vix_percentile = (vix_5y['^VIX'] < current_vix).sum() / len(vix_5y) * 100
        
        logger.info(f"  VOO Price Percentile:    {spy_percentile:.1f}%")
        logger.info(f"  VIX Level Percentile:    {vix_percentile:.1f}%")
        
        if vix_percentile > 70:
            logger.warning(f"  ⚠️  VIX is elevated (top {100-vix_percentile:.0f}% of readings)")
        
        logger.info("-" * 60)
        logger.info("Historical context analysis completed")
            
    except Exception as e:
        logger.error(f"Could not fetch historical context: {e}", exc_info=True)

def update_position(contribution_info, market_data):
    """Update and log historical strategy performance (backtest)"""
    logger.info("\n📊 UPDATING STRATEGY HISTORY (Backtest)...")
    logger.info("-" * 60)
    
    try:
        if not Path("position.csv").exists():
            logger.info("No existing position.csv file found. Created new file.")
            pd.DataFrame(columns=['Date', 'Entry Price', 'Contribution', 'Portfolio Value']).to_csv("position.csv", index=False)
        position = pd.read_csv("position.csv")
        date = datetime.now().strftime('%Y-%m-%d')
        entry_price = round(market_data['current_spy'],2)
        total_contribution = contribution_info['contribution_amount'] + (position['Contribution'].iloc[-1] if not position.empty else 0)
        print(contribution_info['contribution_amount'],position['Portfolio Value'].iloc[-1]/position['Entry Price'].iloc[-1] * market_data['current_spy'])
        portfolio_value = contribution_info['contribution_amount'] + position['Portfolio Value'].iloc[-1]/position['Entry Price'].iloc[-1] * market_data['current_spy']# if not position.empty else contribution_info['contribution_amount']
        no_of_contributions = len(position) + 1
        
        new_entry = pd.DataFrame([{'Date': date, 'Entry Price': entry_price, 'Contribution': total_contribution, 'Portfolio Value': round(portfolio_value,2)}])
        new_entry.to_csv("position.csv", index=False, mode='a', header=False)
        logger.info(f"Strategy history updated: {date}, Entry Price: ${entry_price:.2f}, Total Contribution: ${total_contribution}, Portfolio Value: ${portfolio_value:.2f}, No. of Contributions: {no_of_contributions}")
        
        new_entry ={'Date': date, 'Entry Price': entry_price, 'Contribution': total_contribution, 'Portfolio Value': round(portfolio_value,2), 'No. of Contributions': no_of_contributions}
        
    except Exception as e:
        logger.error(f"Error updating strategy history: {e}", exc_info=True)
        return
    
    return new_entry

# -----------------------------
# Main Function
# -----------------------------
def main():
    """Main execution function"""
    logger.info("\n🚀 Starting SPY Martingale DCA Contribution Calculator...")
    logger.info(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Get current market data
        market_data = get_current_market_data()
        
        # Calculate contribution
        contribution_info = calculate_contribution(market_data)

        # Update position history
        new_entry = update_position(contribution_info, market_data)
        
        # Display notification
        display_notification(market_data, contribution_info, new_entry)

        # Show historical context
        show_historical_context(market_data)
        
        logger.info("\n✅ Analysis complete!")
        logger.info(f"Execution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
    except Exception as e:
        logger.critical(f"Critical error in main execution: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()