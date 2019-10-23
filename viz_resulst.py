import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('D:\QSync\Master_Studium\Semester4\Hauptseminar\Reinforcement_Tutorial\expected_reward3000.csv')

sns.set()
sns.relplot(x='steps', y='expected_reward', data=df, kind='line')
plt.show()
