import pandas as p 
from sim_parameters import TRASITION_PROBS
from sim_parameters import HOLDING_TIMES
import numpy as np
from helper import create_plot as cp
from datetime import datetime
import pandas as pd


def run(countries_csv_name='a3-countries.csv', countries=[], sample_ratio=1e6, start_date='2021-04-01', end_date='2022-04-30'):
    '''we should use the markov chain and plotting here'''
    dfq=p.read_csv(countries_csv_name)

    df = p.DataFrame(dfq)
    #Step 1, we will calculate the sample sizes for each age group
    # Calculate sampling ratios for each age group
    df['(Less_5 Samples)'] = ((df['population'] / int(sample_ratio)) * df['less_5']) / 100
    df['(Less_5 Samples)'] = df['(Less_5 Samples)'].astype(int)

    df['(5_to_14 Samples)'] = (((df['population']/int(sample_ratio))* df['5_to_14'])/100)
    df['(5_to_14 Samples)']= df['(5_to_14 Samples)'].astype(int)

    df['(15 to 24 Samples)'] = (((df['population']/int(sample_ratio)) * df['15_to_24'])/100)
    df['(15 to 24 Samples)'] = df['(15 to 24 Samples)'].astype(int)

    df['(25 to 64 Samples)'] = (((df['population']/int(sample_ratio)) * df['25_to_64'])/100)
    df['(25 to 64 Samples)'] = df['(25 to 64 Samples)'].astype(int)

    df['(over 60 Samples)'] = (((df['population']/int(sample_ratio))* df['over_65'])/100)
    df['(over 60 Samples)'] = df['(over 60 Samples)'].astype(int)

    # Filter the DataFrame using the countries list
    df = df[df['country'].isin(countries)]
    #After this we will drop the columns which are not required
    df.drop(['population','5_to_14','15_to_24','25_to_64','over_65','less_5','median_age'], inplace=True, axis=1)

    # Importantly, we will calculate the sum of the sample sizes for each country
    samplesumlistt=['(Less_5 Samples)','(5_to_14 Samples)','(15 to 24 Samples)','(25 to 64 Samples)','(over 60 Samples)']
    df['sums'] = df[samplesumlistt].sum(axis=1)
    
    # Melt the DataFrame where only the 'country', 'sums', and 'variable' columns are present for next steps
    df = p.melt(df, id_vars=['country', 'sums'])
#############################################################
    #New step 2, Now creating new dataframe with repeated rows based on the sample size in each age group
    # Create a new DataFrame with repeated rows based on the sample size in each age group
    df = df.loc[df.index.repeat(df.iloc[:, 3])]
    
    # Create a new column 'person_id' with a sequence of integers
    df['person_id'] = range(len(df))

    # Create a new column 'age name' based on the column names of the sample sizes
    df['age name'] = df.iloc[:, 2].map({
        "(Less_5 Samples)": "less_5",
        "(5_to_14 Samples)": "5_to_14",
        "(15 to 24 Samples)": "15_to_24",
        "(25 to 64 Samples)": "25_to_64",
        "(over 60 Samples)": "over_65"
    })
    
    # Reseting the index of the DataFrame
    df.reset_index(drop=True, inplace=True)
    # Select only the 'person_id', 'age name', and 'country' columns
    df2 = df[['person_id', 'age name', 'country']]
######################################################

#TASK 2: TO USE THIS simualtion IN OUR DATAFRAME!!!!
    class Simulation:
        def __init__(self, transitions, holding_time):
            self.other_remaining_hours_ = 0
            self.starting_state_ = 'H'  # initial state is healthy
            self.trans_prob_ = transitions
            self.x = holding_time

        def next_state(self):
            if self.other_remaining_hours_ <= 0:
                curr_state = self.starting_state_
                trans_p = self.trans_prob_[curr_state]
                self.other_remaining_hours_ -= 1
                x = list(trans_p.values())
                c = list(trans_p.keys())
                # State gets randomized here
                self.starting_state_ = np.random.choice(c, p=x)
                self.other_remaining_hours_ = self.x[self.starting_state_] - 1
            else:
                self.other_remaining_hours_ -= 1

        def iterable(self):
            while True:
                yield self.starting_state_
                self.next_state()

    # Define the start and end dates
    
    dates = pd.date_range(start_date, end_date)

    # Create a DataFrame to store the results
    timeseries = pd.DataFrame()

    # Iterate over the rows of df2 using iterrows, which is more efficient
    for index, row in df2.iterrows():
        age_group = row['age name']
        transitions = TRASITION_PROBS[age_group]
        holding_time = HOLDING_TIMES[age_group]

        # Create a Simulation object
        sim = Simulation(transitions, holding_time)
        sim_iter = sim.iterable()

        # Generate the states for the given number of days
        states = [next(sim_iter) for _ in range(len(dates))]

        # Create a DataFrame for this person
        df_person = pd.DataFrame({
            'person_id': row['person_id'],
            'age_group': age_group,
            'country': row['country'],
            'date': dates,
            'state': states,
            'staying_days': [holding_time[state] for state in states]
        })

        # Append the DataFrame to the timeseries DataFrame
        timeseries = timeseries._append(df_person)

    # Add the previous state column
    timeseries['prev_stat'] = timeseries['state'].shift()   

  
    timeseries['prev_stat']=timeseries['state']  
    
    #Simulated timeseries is cinverted to csv 
    timeseries.to_csv("a3-covid-simulated-timeseries.csv",index=False)
   
    #EXTRACTING ALL STATES FROM SINGLE STATE COLUMN AND MAKING IT MULTIPLE
    # Copy the DataFrame
    summarydf = timeseries.copy()

    # Extract states from the 'state' column and create new columns for each state
    states = ['H', 'I', 'S', 'M', 'D']
    
    for state in states:
        summarydf[state] = np.where(summarydf['state'].str.contains(state), 1, 0)
    
    # Drop unnecessary columns
    summarydf.drop(['person_id', 'age_group', 'state', 'staying_days', 'prev_stat'], inplace=True, axis=1)

    # Reindex the DataFrame
    new_cols = ["date", "country", "D", "H", "I", "M", "S"]
    summarydf = summarydf.reindex(columns=new_cols)

    # Group by 'date' and 'country' and sum the state columns
    result = summarydf.groupby(['date', 'country'])[states].sum()

    # Write the result to a CSV file
    result.to_csv("a3-covid-summary-timeseries.csv")

    ####################Summary csv complete#########
    #creating plot here!
    cp("a3-covid-summary-timeseries.csv",countries)    