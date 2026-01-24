import pandas as pd
import matplotlib.pyplot as plt

covid = pd.read_csv(r"C:\Users\santh\OneDrive\Desktop\intern\task 4\owid-covid-data.csv")

covid['date'] = pd.to_datetime(covid['date'])

world = covid.groupby('date')[['total_cases','total_deaths','total_vaccinations']].sum()

plt.figure()
plt.plot(world.index, world['total_cases'], label='Total Cases')
plt.plot(world.index, world['total_deaths'], label='Total Deaths')
plt.plot(world.index, world['total_vaccinations'], label='Total Vaccinations')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title('Global COVID-19 Trend')
plt.legend()
plt.show()

top_cases = covid.groupby('location')['total_cases'].max().sort_values(ascending=False).head(10)

plt.figure()
top_cases.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Total Cases')
plt.title('Top 10 Countries by COVID Cases')
plt.show()

top_deaths = covid.groupby('location')['total_deaths'].max().sort_values(ascending=False).head(10)

plt.figure()
top_deaths.plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Total Deaths')
plt.title('Top 10 Countries by COVID Deaths')
plt.show()
