import pandas as pd
import dask.dataframe as dd
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday
from pandas.tseries.offsets import Day
from dateutil.relativedelta import MO, TH

class BusyHolidayTravelCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('New Years Day', month=1, day=1),
        Holiday('New Years Eve', month=12, day=31),
        Holiday("Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))),
        Holiday('Friday Before Memorial Day', month=5, day=31, offset=[pd.DateOffset(weekday=MO(-1)), Day(-3)]),
        Holiday('IndependenceDay', month=7, day=4),
        Holiday("Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))),
        Holiday("Friday Before Labor Day", month=9, day=1, offset=[pd.DateOffset(weekday=MO(1)), Day(-3)]),
        Holiday("Thanksgiving Day", month=11, day=1, offset=pd.DateOffset(weekday=TH(4))),
        Holiday('Christmas', month=12, day=25),
        Holiday('Christmas Eve', month=12, day=24)
    ]

pd.options.display.max_columns = None
pd.options.display.max_rows = None

ddf_2019 = dd.read_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/2019.csv', low_memory=False, dtype={'CancellationCode': 'object',
       'Div1Airport': 'object',
       'Div1TailNum': 'object',
       'Div2Airport': 'object',
       'Div2TailNum': 'object',
       'Div3Airport': 'object',
       'Div3TailNum': 'object',
       'CRSDepTime': 'object',
       'CRSArrTime': 'object'})
ddf_2020 = dd.read_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/2020.csv', low_memory=False, dtype={'CancellationCode': 'object',
       'Div1Airport': 'object',
       'Div1TailNum': 'object',
       'Div2Airport': 'object',
       'Div2TailNum': 'object',
       'Div3Airport': 'object',
       'CRSDepTime': 'object',
       'CRSArrTime': 'object'})
ddf_2021 = dd.read_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/2021.csv', low_memory=False, dtype={'CancellationCode': 'object',
       'Div1Airport': 'object',
       'Div1TailNum': 'object',
       'Div2Airport': 'object',
       'Div2TailNum': 'object',
       'Div3Airport': 'object',
       'CRSDepTime': 'object',
       'CRSArrTime': 'object'})
ddf_2022 = dd.read_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/2022.csv', low_memory=False, dtype={'CancellationCode': 'object',
       'Div1Airport': 'object',
       'Div1TailNum': 'object',
       'Div2Airport': 'object', 
       'Div2TailNum': 'object',
       'Div3Airport': 'object',
       'CRSDepTime': 'object',
       'CRSArrTime': 'object'})

ddf_2022 = ddf_2022.drop(['IATA_CODE_Reporting_Airline', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin', 'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportSeqID', 'DestCityMarketID', 'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestCityMarketID', 'DestStateName', 'DestWac', 'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID', \
        'Div1WheelsOn', 'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum', 'Div2Airport', 'Div2AirportID', 'Div2AirportSeqID', \
        'Div2WheelsOn', 'Div2TotalGTime', 'Div2LongestGTime', 'Div2WheelsOff', 'Div2TailNum', 'Div3Airport', 'Div3AirportID', 'Div3AirportSeqID', \
        'Div3WheelsOn', 'Div3TotalGTime', 'Div3LongestGTime', 'Div3WheelsOff', 'Div3TailNum', 'Div4Airport', 'Div4AirportID', 'Div4AirportSeqID', \
        'Div4WheelsOn', 'Div4TotalGTime', 'Div4LongestGTime', 'Div4WheelsOff', 'Div4TailNum', 'Div5Airport', 'Div5AirportID', 'Div5AirportSeqID', \
        'Div5WheelsOn', 'Div5TotalGTime', 'Div5LongestGTime', 'Div5WheelsOff', 'Div5TailNum'], axis=1)

ddf_2021 = ddf_2021.drop(['IATA_CODE_Reporting_Airline', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin', 'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportSeqID', 'DestCityMarketID', 'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestCityMarketID', 'DestStateName', 'DestWac', 'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID', \
        'Div1WheelsOn', 'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum', 'Div2Airport', 'Div2AirportID', 'Div2AirportSeqID', \
        'Div2WheelsOn', 'Div2TotalGTime', 'Div2LongestGTime', 'Div2WheelsOff', 'Div2TailNum', 'Div3Airport', 'Div3AirportID', 'Div3AirportSeqID', \
        'Div3WheelsOn', 'Div3TotalGTime', 'Div3LongestGTime', 'Div3WheelsOff', 'Div3TailNum', 'Div4Airport', 'Div4AirportID', 'Div4AirportSeqID', \
        'Div4WheelsOn', 'Div4TotalGTime', 'Div4LongestGTime', 'Div4WheelsOff', 'Div4TailNum', 'Div5Airport', 'Div5AirportID', 'Div5AirportSeqID', \
        'Div5WheelsOn', 'Div5TotalGTime', 'Div5LongestGTime', 'Div5WheelsOff', 'Div5TailNum'], axis=1)

ddf_2020 = ddf_2020.drop(['IATA_CODE_Reporting_Airline', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin', 'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportSeqID', 'DestCityMarketID', 'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestCityMarketID', 'DestStateName', 'DestWac', 'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID', \
        'Div1WheelsOn', 'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum', 'Div2Airport', 'Div2AirportID', 'Div2AirportSeqID', \
        'Div2WheelsOn', 'Div2TotalGTime', 'Div2LongestGTime', 'Div2WheelsOff', 'Div2TailNum', 'Div3Airport', 'Div3AirportID', 'Div3AirportSeqID', \
        'Div3WheelsOn', 'Div3TotalGTime', 'Div3LongestGTime', 'Div3WheelsOff', 'Div3TailNum', 'Div4Airport', 'Div4AirportID', 'Div4AirportSeqID', \
        'Div4WheelsOn', 'Div4TotalGTime', 'Div4LongestGTime', 'Div4WheelsOff', 'Div4TailNum', 'Div5Airport', 'Div5AirportID', 'Div5AirportSeqID', \
        'Div5WheelsOn', 'Div5TotalGTime', 'Div5LongestGTime', 'Div5WheelsOff', 'Div5TailNum'], axis=1)

"""ddf_2019 = ddf_2019.drop(['IATA_CODE_Reporting_Airline', 'OriginAirportSeqID', 'OriginCityMarketID', 'Origin', 'OriginCityName', 'OriginState', 'OriginStateFips', 'OriginStateName', 'OriginWac', 'DestAirportSeqID', 'DestCityMarketID', 'Dest', 'DestCityName', 'DestState', 'DestStateFips', 'DestCityMarketID', 'DestStateName', 'DestWac', 'Div1Airport', 'Div1AirportID', 'Div1AirportSeqID', \
        'Div1WheelsOn', 'Div1TotalGTime', 'Div1LongestGTime', 'Div1WheelsOff', 'Div1TailNum', 'Div2Airport', 'Div2AirportID', 'Div2AirportSeqID', \
        'Div2WheelsOn', 'Div2TotalGTime', 'Div2LongestGTime', 'Div2WheelsOff', 'Div2TailNum', 'Div3Airport', 'Div3AirportID', 'Div3AirportSeqID', \
        'Div3WheelsOn', 'Div3TotalGTime', 'Div3LongestGTime', 'Div3WheelsOff', 'Div3TailNum', 'Div4Airport', 'Div4AirportID', 'Div4AirportSeqID', \
        'Div4WheelsOn', 'Div4TotalGTime', 'Div4LongestGTime', 'Div4WheelsOff', 'Div4TailNum', 'Div5Airport', 'Div5AirportID', 'Div5AirportSeqID', \
        'Div5WheelsOn', 'Div5TotalGTime', 'Div5LongestGTime', 'Div5WheelsOff', 'Div5TailNum'], axis=1)"""

#df_2019 = ddf_2019.compute()
df_2020 = ddf_2020.compute()
df_2021 = ddf_2021.compute()
df_2022 = ddf_2022.compute()

#combined_df = pd.concat([df_2019, df_2020, df_2021, df_2022])
combined_df = pd.concat([df_2020, df_2021, df_2022])

combined_df = combined_df.drop(['DivReachedDest', 'DivActualElapsedTime', 'DivArrDelay', 'DivDistance', 'DivAirportLandings', 'TotalAddGTime', 'LongestAddGTime', 'TaxiOut', 'WheelsOff', 'WheelsOn', 'TaxiIn'], axis=1)

combined_df['WeatherDelay'] = combined_df['WeatherDelay'].fillna(0)
combined_df['NASDelay'] = combined_df['NASDelay'].fillna(0)
combined_df['SecurityDelay'] = combined_df['SecurityDelay'].fillna(0)
combined_df['LateAircraftDelay'] = combined_df['LateAircraftDelay'].fillna(0)
combined_df['CarrierDelay'] = combined_df['CarrierDelay'].fillna(0)

combined_df['OnTime'] = 1

combined_df.loc[(combined_df['Cancelled'] > 0), 'OnTime'] = 0
combined_df.loc[(combined_df['Diverted'] > 0), 'OnTime'] = 0
combined_df.loc[(combined_df['DepDel15'] > 0), 'OnTime'] = 0

#All values the same
combined_df = combined_df.drop(['Flights'], axis=1)

combined_df['CancelledDueToWeather'] = 0
combined_df['CancelledDueToCarrier'] = 0
combined_df['CancelledDueToNAS'] = 0
combined_df['CancelledDueToSecurity'] = 0

combined_df.loc[(combined_df['CancellationCode'] == 'A'), 'CancelledDueToCarrier'] = 1
combined_df.loc[(combined_df['CancellationCode'] == 'B'), 'CancelledDueToWeather'] = 1
combined_df.loc[(combined_df['CancellationCode'] == 'C'), 'CancelledDueToNAS'] = 1
combined_df.loc[(combined_df['CancellationCode'] == 'D'), 'CancelledDueToSecurity'] = 1

combined_df['FlightDate'] = combined_df['FlightDate'].astype('datetime64')

#calculate holiday
dr = pd.date_range(start='2019-01-01', end='2022-08-31')
calendar = BusyHolidayTravelCalendar()
holidays = calendar.holidays(start=dr.min(), end=dr.max())
combined_df['Holiday'] = combined_df['FlightDate'].isin(holidays)

combined_df = combined_df.drop(['CancellationCode', 'FlightDate'], axis=1)

#Split out the hour of the scheduled departure/arrival to store in a separate feature
combined_df['DepartureHour'] = combined_df['CRSDepTime'].str[:2]
combined_df['DepartureHour'] = combined_df['DepartureHour'].astype(int)
combined_df['ArrivalHour'] = combined_df['CRSArrTime'].str[:2]
combined_df['ArrivalHour'] = combined_df['ArrivalHour'].astype(int)

combined_df.to_csv('/Users/jessicahalterman/Documents/MachineLearningProject/Data/allData2.csv', index=False)