import telegram
import config
import asyncio
from datetime import datetime, timedelta
import pandas as pd

async def send_to_channel(multiplier, new_entry):
    time = datetime.now()# + timedelta(hours=8)  # Adjust to your timezone

    bot = telegram.Bot(token=config.API_KEY)

    rec_msg = f"💰 Contribution Recommendation: \n"
    rec_msg += "(Contribute $1000 every 1st and 3rd Mon in each Month) \n"
    rec_msg += f"Multiplier: {multiplier}x \n"

    entry_price = new_entry['Entry Price']
    contribution = new_entry['Contribution']
    portfolio_value = new_entry['Portfolio Value']
    no_of_contributions = new_entry['No. of Contributions']

    rec_msg += "-" * 40 + "\n"
    rec_msg += f"💵 Current Portfolio: \n"
    rec_msg += f"Entry Price: ${entry_price:.2f} \n"
    rec_msg += f"Total Contribution: ${contribution} \n"
    rec_msg += f"Total Portfolio Value: ${portfolio_value} \n"
    rec_msg += f"No. of Contributions: {no_of_contributions} \n"

    message = time.strftime("%Y%m%d (%a)") + f""":\n\n{rec_msg}\n以上內容只供參考，並非投資意見，亦不構成任何投資產品之要約、要約招攬或建議。投資附帶風險，過往表現並不代表將來的表現。本資料只作為一般用途，它並沒有考慮您的個人需要、投資目標及特定財政狀況。"""
    await bot.send_message(chat_id=config.CHANNEL_ID, text=message)

# asyncio.run(send_to_channel(1))
