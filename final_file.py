import pandas as pd

#First I need to load the csvs into a dataframe and perform initial cleaning
#I will also immediately save these cleaned dataframes into 3 json files.
def data_load_from_csv_to_json_and_clean():
    activity_logs_dataframe = pd.read_csv("ACTIVITY_LOG.csv")
    component_codes_dataframe = pd.read_csv("COMPONENT_CODES.csv")
    user_log_dataframe = pd.read_csv("USER_LOG.csv")

    #Cleaning techniques below:
    #strip whitespace for all files.
    activity_logs_dataframe.columns = activity_logs_dataframe.columns.str.strip()
    component_codes_dataframe.columns = component_codes_dataframe.columns.str.strip()
    user_log_dataframe.columns = user_log_dataframe.columns.str.strip()

    #Activity log cleaning specifically below:
    #for activity logs, fill in 'unknown' for each key, or 0 for the user's ID.
    activity_logs_dataframe.fillna({'Component': 'Unknown', 'Action': 'Unknown', 'Target': 'Unknown',
                                    'User Full Name *Anonymized': 0})
    
    #Removing underscore from target column
    activity_logs_dataframe['Target'] = activity_logs_dataframe['Target'].str.replace('_', '', 
                                                                                      regex=False)
    
    #Removing underscore from component column
    activity_logs_dataframe['Component'] = activity_logs_dataframe['Component'].str.replace('_', '', 
                                                                                      regex=False)
    #User log cleaning specifically below:
    #remove redundant time part from the Date column and convert to datetime format
    #I am doing this as the '00:00' in the date key-value pair was adding no value in the data
    user_log_dataframe['Date'] = pd.to_datetime(user_log_dataframe['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    user_log_dataframe['Date'] = user_log_dataframe['Date'].dt.strftime('%d/%m/%Y')

    #filling in missing date values with Christmas 2023 as a default value
    user_log_dataframe['Date'].fillna('25/11/2023', inplace=True)

    #Strip whitespace in the Time column, almost every time value had a ' ' at the start
    user_log_dataframe['Time'] = user_log_dataframe['Time'].str.strip()

    #Filling in missing time values with a default time value
    user_log_dataframe['Time'].fillna('00:00:00', inplace=True)

    # Remove non-alphanumeric characters (if needed) - I MIGHT GO BACK IN AND STRIP THIS
    #user_log_dataframe['Time'] = user_log_dataframe['Time'].str.replace(r'\W', '', regex=True)

    #Component_codes cleaning below:
    #Removing underscore from component column
    component_codes_dataframe['Component'] = component_codes_dataframe['Component'].str.replace('_', '', 
                                                                                      regex=False)
    #Removing underscore from code column
    component_codes_dataframe['Code'] = component_codes_dataframe['Code'].str.replace('_', '', 
                                                                                      regex=False)
    
    """activity_logs_dataframe.to_json("activity_logs.json", orient='records', lines=False, indent=4)
    component_codes_dataframe.to_json("component_codes.json", orient='records', lines=False, indent=4)
    user_log_dataframe.to_json("user_logs.json", orient='records', lines=False, indent=4)"""

    return activity_logs_dataframe, component_codes_dataframe, user_log_dataframe

activity_logs_dataframe, component_codes_dataframe, user_log_dataframe = data_load_from_csv_to_json_and_clean()

#Task 1, remove any output of the component column containing system or folder:
def remove(activity_logs_dataframe, component_codes_dataframe):
    activity_logs_dataframe = activity_logs_dataframe[~activity_logs_dataframe['Component'].isin(['System', 'Folder'])]
    component_codes_dataframe = component_codes_dataframe[~component_codes_dataframe['Component'].isin(['System', 'Folder'])]
    return activity_logs_dataframe, component_codes_dataframe

#Task 2, rename 'User Full Name *Anonymized' to user_ID across activity logs and user logs
def rename(activity_logs_dataframe, user_log_dataframe):
    activity_logs_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)
    user_log_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)

    return activity_logs_dataframe, user_log_dataframe

remove(activity_logs_dataframe, component_codes_dataframe)
rename(activity_logs_dataframe, user_log_dataframe)
#Task 3: merge all 3 CSV files together.. 
def merge_data_to_one_frame(activity_logs_dataframe, component_codes_dataframe, user_log_dataframe):
    #PRINT OUT TO THE USER VIA MESSAGE BOX SOMETHING LIKE:
    #"THIS PROCESS WILL TAKE A WHILE, PLEASE BARE WITH US WHILST WE MERGE THE DATA."
    first_merge = pd.merge(activity_logs_dataframe, component_codes_dataframe, 
                                         on='Component', how='left')
    full_merge = pd.merge(first_merge, user_log_dataframe, 
                                 on='User_ID', how='left')
    
    return full_merge
    

fully_merged_dataset = merge_data_to_one_frame(activity_logs_dataframe, component_codes_dataframe, user_log_dataframe)
print(fully_merged_dataset.head())

#Task 4 and 5, reshape the data using pivot, I am reshaping in advance due to the 
#statistical analysis that will be required in the further tasks.

def reshape(fully_merged_dataset):
    #First, string slice to get the month.
    fully_merged_dataset['Month'] = fully_merged_dataset['Date'].str[3:5]

    #interaction_count is Task 5, this new column counts each interaction for each user_Id in each
    #month, creating new columns like Component_10 (October's count for that unique component)
    #This counts for how many times that same component has come up so far for that user_ID
    interaction_count = fully_merged_dataset.groupby(['User_ID', 
                                                     'Component', 'Month']
                                                     ).size().reset_index(name='Interaction_Count')
    
    # Pivot the data
    reshaped_data = pd.pivot_table(
        interaction_count, 
        index=['User_ID'],  # User_ID as index
        columns=['Component', 'Month'],  # Columns will be Component and Month
        values='Interaction_Count',  # Values to count interactions
        aggfunc='sum',  # Sum the interactions for the same user and component
        fill_value=0  # Fill NaN with 0 where there were no interactions
    )

    #Reset index for readability
    reshaped_data.reset_index(inplace=True)

    # Display the reshaped data
    print(reshaped_data.shape)
    return reshaped_data

reshape(fully_merged_dataset)
