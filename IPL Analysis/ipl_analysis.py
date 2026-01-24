import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ipl = pd.read_csv(r"C:\Users\santh\OneDrive\Desktop\intern\task 3\matches.csv")

print(ipl.head())
print(ipl.info())
print(ipl.describe())
print(ipl.isnull().sum())

team_wins = ipl['winner'].value_counts()
print(team_wins)

plt.figure(figsize=(10,5))
sns.barplot(x=team_wins.index, y=team_wins.values, palette="viridis")
plt.xticks(rotation=45)
plt.title("Most Winning IPL Teams")
plt.ylabel("Number of Wins")
plt.show()
venue_counts = ipl['venue'].value_counts().head(10)
plt.figure(figsize=(10,5))
sns.barplot(x=venue_counts.index, y=venue_counts.values, palette="coolwarm")
plt.xticks(rotation=90)
plt.title("Top 10 IPL Venues")
plt.ylabel("Number of Matches")
plt.show()
plt.figure(figsize=(12,6))
sns.countplot(data=ipl, x='season', hue='winner', palette="tab20")
plt.xticks(rotation=45)
plt.title("Season-wise Winning Teams")
plt.ylabel("Number of Matches Won")
plt.show()
