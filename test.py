import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import quandl
quandl.ApiConfig.api_key = "g-jhivSzUeDwajNsoqKd"



aluminum_shanghai = 'CHRIS/SHFE_AL'
gold_shanghai = 'CHRIS/SHFE_AU'
soybean_oil = 'CHRIS/CME_BO'
corn = 'CHRIS/CME_C'
canadian_dollar = 'CHRIS/CME_CD'
copper_shanghai = 'CHRIS/SHFE_CU'
dax_futures = 'CHRIS/EUREX_FDAX'
palladium = 'CHRIS/CME_PA'
ten_year_treasury = 'CHRIS/CME_TY'




today = date.today()
one_year_ago = today - relativedelta(years=1)
ten_years_ago = today - relativedelta(years=10)


def get_prices(index, depth, start, end):

	data_prices = quandl.get(
		index + str(depth), 
		start_date = start, 
		end_date = end)

	data_prices.index = pd.to_datetime(data_prices.index)

	return data_prices



train_data = get_prices(gold_shanghai, 1, ten_years_ago, one_year_ago)
test_data = get_prices(gold_shanghai, 1, one_year_ago, today)




