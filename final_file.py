import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

     # Get the first 10 rows
    """reshaped_data_first_10 = reshaped_data.head(10)

    # Save the first 10 rows to JSON
    reshaped_data_first_10.to_json("reshaped_data_first_10.json", orient='split', indent=4)"""
    # Display the reshaped data
    print(reshaped_data.shape)
    return reshaped_data

reshaped_data = reshape(fully_merged_dataset)

"""#Task 6, output statistics

def output_statistics(reshaped_data):
    print("Reshaped Data (Head):")
    print(reshaped_data.head())
    print("\nReshaped Data Columns:")
    print(reshaped_data.columns)
    
    # Specify the components you're interested in
    components_of_interest = ['Quiz', 'Lecture', 'Assignment', 'Attendence', 'Survey']
    
    # Filter the reshaped_data to only include those components
    relevant_columns = [
        col for col in reshaped_data.columns 
        if col[0] in components_of_interest  # Use tuple index for component check
    ]
    
    # Calculate statistics for each month (September, October, November, December)
    monthly_statistics = {}
    for month in ['09', '10', '11', '12']:  # September, October, November, December
        # Filter columns for the current month
        month_columns = [col for col in relevant_columns if col[1] == month]
        
        # Select the columns for the current month
        month_data = reshaped_data.loc[:, [('User_ID', '')] + month_columns]
        
        # Initialize dictionary for month stats
        monthly_statistics[month] = {}
        
        # Mean for this month
        monthly_statistics[month]['Mean'] = month_data[month_columns].mean()
        
        # Median for this month
        monthly_statistics[month]['Median'] = month_data[month_columns].median()
        
        # Mode for this month (returning the first mode for each component)
        monthly_statistics[month]['Mode'] = month_data[month_columns].mode().iloc[0]  # Take the first mode value

    # Print out the statistics per month
    print("Statistics per month:")
    for month, stats in monthly_statistics.items():
        print(f"\nMonth: {month}")
        print("Mean:\n", stats['Mean'])
        print("Median:\n", stats['Median'])
        print("Mode:\n", stats['Mode'])

    # Calculate overall semester statistics (September to December)
    semester_statistics = {}
    
    for component in components_of_interest:
        # Filter columns for the current component across all months
        component_columns = [col for col in relevant_columns if col[0] == component]
        
        # Select the columns for the current component
        component_data = reshaped_data.loc[:, [('User_ID', '')] + component_columns]
        
        # Calculate Mean for the entire semester (across all months)
        semester_statistics[component] = {}
        semester_statistics[component]['Mean'] = component_data[component_columns].mean().mean()
        
        # Calculate Median for the entire semester (across all months)
        semester_statistics[component]['Median'] = component_data[component_columns].median().median()
        
        # Calculate Mode for the entire semester (across all months)
        # Concatenate data from all months for the component and calculate mode
        combined_data = component_data[component_columns].stack()  # Stack all the month data into a single column
        semester_statistics[component]['Mode'] = combined_data.mode().iloc[0]  # Take the first mode value from the combined data
    
    # Print out the overall semester statistics
    print("\nStatistics for the entire 13-week semester (September to December):")
    for component, stats in semester_statistics.items():
        print(f"\nComponent: {component}")
        print("Mean:\n", stats['Mean'])
        print("Median:\n", stats['Median'])
        print("Mode:\n", stats['Mode'])
    
    return monthly_statistics, semester_statistics

# Assuming reshaped_data is already available after the pivot
output_statistics(reshaped_data)

#Task 7, I split this into two seperate functions, one for the initial graphs,
#then the second one with calculating the correlation and displaying it.

# Plot Component Interactions (Task 6)
def plot_component_interactions(reshaped_data):
    components_of_interest = ['Assignment', 'Quiz', 'Lecture', 'Book', 'Project', 'Course']
    fig, axes = plt.subplots(len(components_of_interest), 1, figsize=(10, 12))
    
    # Flatten the column headers from multi-level index
    reshaped_data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in reshaped_data.columns]
    
    for idx, component in enumerate(components_of_interest):
        # Select columns related to the current component
        component_columns = [col for col in reshaped_data.columns if col.startswith(component)]
        
        # Sum across the selected columns for each month
        component_data = reshaped_data[component_columns].sum(axis=0)
        
        # Plot the data
        axes[idx].bar(component_data.index, component_data.values)
        axes[idx].set_title(f'User Interactions with {component}')
        axes[idx].set_xlabel('Month')
        axes[idx].set_ylabel('Total Interactions')
        axes[idx].tick_params(axis='x', rotation=45)  # Rotate x-axis labels for readability
    
    plt.tight_layout()
    plt.show()


# Calculate and Plot Correlation using a Heatmap (Task 7)
def calculate_and_plot_correlation(reshaped_data):
    correlation_matrix = reshaped_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Matrix between Components")
    plt.show()

# Visualize component interactions
plot_component_interactions(reshaped_data)

# Calculate and plot correlation
calculate_and_plot_correlation(reshaped_data)"""

def plot_component_interactions_separately(reshaped_data):
    components_of_interest = ['Assignment', 'Quiz', 'Lecture', 'Book', 'Project', 'Course']
    
    # Flatten the column headers to make it easier to filter by component and month
    reshaped_data = reshaped_data.copy()
    reshaped_data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in reshaped_data.columns]
    
    for component in components_of_interest:
        # Select columns related to the current component
        component_columns = [col for col in reshaped_data.columns if col.startswith(component)]
        
        # Sum across the selected columns for each month
        component_data = reshaped_data[component_columns].sum(axis=0)
        
        # Create a new figure for each component
        plt.figure(figsize=(8, 6))
        plt.bar(component_data.index, component_data.values)
        plt.title(f'User Interactions with {component}')
        plt.xlabel('Month')
        plt.ylabel('Total Interactions')
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability
        
        # Show the plot before creating the next one
        plt.tight_layout()
        plt.show()

def plot_user_component_interactions(reshaped_data):
    # Set 'User_ID' as the index
    interaction_data = reshaped_data.set_index(('User_ID', ''))

    # Flatten columns to make it easier to work with component names and months
    interaction_data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in interaction_data.columns]
    
    # Plot the heatmap of user interactions with components
    plt.figure(figsize=(12, 8))
    sns.heatmap(interaction_data, annot=False, cmap='YlGnBu', linewidths=0.5)
    plt.title("User Interactions with Components")
    plt.xlabel("Component_Month")
    plt.ylabel("User_ID")
    plt.show()

plot_component_interactions_separately(reshaped_data)
plot_user_component_interactions(reshaped_data)