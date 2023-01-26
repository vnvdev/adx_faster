import adx
import pandas as pd

df = pd.read_csv("youdata.csv")
df['adx'] = adx.ADX(df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy(), 7) #You can change 7 to any lenght you want
#This functions calculate adx same mt5
