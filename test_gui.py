import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.stats import pearsonr, spearmanr

sns.set(style="whitegrid")

# Initialize main data variables
activity_logs_dataframe = None
component_codes_dataframe = None
user_log_dataframe = None
fully_merged_dataset = None
reshaped_data = None

# Function to load and clean data from CSV files
def data_load_from_csv_to_json_and_clean():
    global activity_logs_dataframe, component_codes_dataframe, user_log_dataframe
    file_paths = filedialog.askopenfilenames(title="Select CSV files", filetypes=[("CSV files", "*.csv")])
    if len(file_paths) != 3:
        messagebox.showerror("Error", "Please select exactly three CSV files.")
        return

    # Loading CSVs
    activity_logs_dataframe = pd.read_csv(file_paths[0])
    component_codes_dataframe = pd.read_csv(file_paths[1])
    user_log_dataframe = pd.read_csv(file_paths[2])

    # Cleaning data
    activity_logs_dataframe.columns = activity_logs_dataframe.columns.str.strip()
    component_codes_dataframe.columns = component_codes_dataframe.columns.str.strip()
    user_log_dataframe.columns = user_log_dataframe.columns.str.strip()

    activity_logs_dataframe.fillna({'Component': 'Unknown', 'Action': 'Unknown', 'Target': 'Unknown', 'User Full Name *Anonymized': 0}, inplace=True)
    activity_logs_dataframe['Target'] = activity_logs_dataframe['Target'].str.replace('_', '', regex=False)
    activity_logs_dataframe['Component'] = activity_logs_dataframe['Component'].str.replace('_', '', regex=False)
    activity_logs_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)
    # Filter out "System" and "Folder" components
    activity_logs_dataframe = activity_logs_dataframe[~activity_logs_dataframe['Component'].isin(['System', 'Folder'])]

    user_log_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)
    user_log_dataframe['Date'] = pd.to_datetime(user_log_dataframe['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    user_log_dataframe['Date'] = user_log_dataframe['Date'].dt.strftime('%d/%m/%Y')
    user_log_dataframe['Date'].fillna('25/11/2023', inplace=True)
    user_log_dataframe['Time'] = user_log_dataframe['Time'].str.strip()
    user_log_dataframe['Time'].fillna('00:00:00', inplace=True)

    component_codes_dataframe['Component'] = component_codes_dataframe['Component'].str.replace('_', '', regex=False)
    component_codes_dataframe['Code'] = component_codes_dataframe['Code'].str.replace('_', '', regex=False)

    messagebox.showinfo("Info", "Data loaded and cleaned successfully.")

# Function to merge data
def merge_data():
    global fully_merged_dataset
    if activity_logs_dataframe is None or component_codes_dataframe is None or user_log_dataframe is None:
        messagebox.showerror("Error", "Please load and clean the data first.")
        return
    messagebox.showinfo("Info", "This process may take up to 5-10 minutes. Please wait.")
    first_merge = pd.merge(activity_logs_dataframe, component_codes_dataframe, on='Component', how='left')
    fully_merged_dataset = pd.merge(first_merge, user_log_dataframe, on='User_ID', how='left')
    messagebox.showinfo("Info", "Data merged successfully.")

# Function to reshape data
def reshape_data():
    global reshaped_data
    if fully_merged_dataset is None:
        messagebox.showerror("Error", "Please merge the data first.")
        return
    messagebox.showinfo("Info", "This process may take up to 5-10 minutes. Please wait.")
    
    fully_merged_dataset['Month'] = fully_merged_dataset['Date'].str[3:5]
    interaction_count = fully_merged_dataset.groupby(['User_ID', 'Component', 'Month']).size().reset_index(name='Interaction_Count')
    reshaped_data = pd.pivot_table(
        interaction_count,
        index=['User_ID'],
        columns=['Component', 'Month'],
        values='Interaction_Count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    messagebox.showinfo("Info", "Data reshaped successfully.")

def output_statistics():
    """
    Output monthly and semester-level statistics for key components.
    """
    global reshaped_data
    
    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    # Make a copy of reshaped_data to avoid modifying the original for visualization
    manipulated_data = reshaped_data.copy()

    # Components of interest
    components_of_interest = ['Quiz', 'Lecture', 'Assignment', 'Attendance', 'Survey']
    
    # Flatten columns to access them easily in the copy (not modifying the original reshaped_data)
    manipulated_data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in manipulated_data.columns]

    # Get columns that contain component-month pairs
    relevant_columns = [col for col in manipulated_data.columns if any(comp in col for comp in components_of_interest)]

    # Monthly statistics (September, October, November, December)
    monthly_statistics = {}
    for month in ['09', '10', '11', '12']:
        month_columns = [col for col in relevant_columns if f'_{month}' in col]
        month_data = manipulated_data[month_columns]

        # Initialize dictionary for month stats
        monthly_statistics[month] = {}
        
        # Mean for this month
        monthly_statistics[month]['Mean'] = month_data.mean().round(2)
        
        # Median for this month
        monthly_statistics[month]['Median'] = month_data.median().to_dict()
        
        # Mode for this month (returning the first mode for each component)
        mode_values = month_data.mode()
        if not mode_values.empty:
            monthly_statistics[month]['Mode'] = mode_values.iloc[0].to_dict()  # Mode per component
        else:
            monthly_statistics[month]['Mode'] = 'No mode'  # Handle the case where there's no mode

    # Semester statistics (September to December combined)
    semester_statistics = {}
    for component in components_of_interest:
        # Filter columns for the current component across all months
        component_columns = [col for col in relevant_columns if col.startswith(component)]
        
        # Select the columns for the current component
        component_data = manipulated_data[component_columns]
        
        # Calculate Mean for the entire semester (across all months)
        semester_statistics[component] = {}
        semester_statistics[component]['Mean'] = component_data.mean().mean()  # Mean of all months combined
        
        # Calculate Median for the entire semester (across all months)
        semester_statistics[component]['Median'] = component_data.median().median()  # Median of all months combined
        
        # Calculate Mode for the entire semester (across all months)
        # Combine data from all months for the component and calculate mode
        combined_data = component_data.stack()  # Stack all the month data into a single column
        mode_values = combined_data.mode()
        if not mode_values.empty:
            semester_statistics[component]['Mode'] = mode_values.iloc[0]  # First mode value for the component
        else:
            semester_statistics[component]['Mode'] = 'No mode'  # Handle the case where there's no mode
    
    # Prepare output message for monthly statistics
    stats_message = "Monthly statistics:\n"
    for month, stats in monthly_statistics.items():
        stats_message += f"\nMonth: {month}\nMean:\n{stats['Mean']}\nMedian:\n{stats['Median']}\nMode:\n{stats['Mode']}\n"
    
    # Prepare output message for semester statistics
    stats_message += "\nSemester statistics:\n"
    for component, stats in semester_statistics.items():
        stats_message += f"\nComponent: {component}\nMean: {stats['Mean']}\nMedian: {stats['Median']}\nMode: {stats['Mode']}\n"
    
    # Print statistics to the console
    print(stats_message)

    # Display a message box to inform the user that statistics have been printed to the console
    messagebox.showinfo("Info", "Statistics have been printed to the console, please check.")

def plot_component_interactions_separately():
    global reshaped_data
    
    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    components_of_interest = ['Assignment', 'Quiz', 'Lecture', 'Book', 'Project', 'Course']
    
    # Make a copy of reshaped_data to avoid modifying the original
    plot_data = reshaped_data.copy()
    
    # Convert column tuples to strings with underscores for month tracking
    plot_data.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in plot_data.columns]

    for component in components_of_interest:
        # Get columns for this component
        component_columns = [col for col in plot_data.columns if col.startswith(component) and '_' in col]
        
        if component_columns:  # Only plot if we have data for this component
            # Sum interactions across all users for each month
            component_data = plot_data[component_columns].sum()
            
            # Create bar plot
            plt.figure(figsize=(8, 6))
            plt.bar(component_data.index, component_data.values)
            plt.title(f'User Interactions with {component}')
            plt.xlabel('Month')
            plt.ylabel('Total Interactions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

"""#HEATMAPS/CORRELATION
def plot_component_correlation():
    global reshaped_data

    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    # Make a copy of reshaped_data to avoid modifying the original
    correlation_data = reshaped_data.copy()

    # Flatten column names to make component-month pairs easier to identify
    correlation_data.columns = ['_'.join(map(str, col)) for col in correlation_data.columns]
    
    # Select columns for components of interest only
    components_of_interest = ['Assignment', 'Quiz', 'Lecture', 'Book', 'Project', 'Course']
    relevant_columns = [col for col in correlation_data.columns if any(comp in col for comp in components_of_interest)]
    
    # Filter the reshaped data to include only relevant component-month columns
    correlation_data = correlation_data[relevant_columns]
    
    # Calculate the correlation matrix for the selected components
    correlation_matrix = correlation_data.corr()

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
    plt.title("Correlation between Components based on User Interactions")
    plt.xlabel("Components")
    plt.ylabel("Components")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

#VISUAL ONLY HEATMAP
def plot_user_component_interactions():
    global reshaped_data

    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    # Flatten the column names for easier access
    reshaped_data.columns = ['_'.join(map(str, col)) for col in reshaped_data.columns]

    # Prepare the data for the heatmap
    interaction_data = reshaped_data.copy()

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(interaction_data, annot=False, cmap='YlGnBu', linewidths=0.5)
    plt.title("User Interactions with Components")
    plt.xlabel("Component_Month")
    plt.ylabel("User_ID")
    plt.show()"""

def plot_user_component_correlation():
    global reshaped_data

    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    # Make a copy of reshaped_data to avoid modifying the original
    correlation_data = reshaped_data.copy()

    # Set the index with 'User_ID' and a placeholder empty string to match your suggestion
    correlation_data = correlation_data.set_index(('User_ID', ''))

    # Flatten column names to make component-month pairs easier to identify
    correlation_data.columns = ['_'.join(map(str, col)) for col in correlation_data.columns]

    # Extract User_ID from the first level of the index (position 0)
    user_id_data = correlation_data.index.get_level_values(0)  # Access the first level of the index

    # Select columns for components (excluding 'User_ID')
    component_columns = [col for col in correlation_data.columns if 'User_ID' not in col]

    # Initialize dictionary to store results
    correlation_results = {}

    # Loop through the components to calculate correlations
    for component in component_columns:
        # Pearson correlation with User_ID
        pearson_corr, pearson_p = pearsonr(correlation_data[component], user_id_data)

        # Spearman correlation with User_ID
        spearman_corr, spearman_p = spearmanr(correlation_data[component], user_id_data)

        # Save correlation results
        correlation_results[component] = {
            'Pearson Correlation': pearson_corr,
            'Pearson p-value': pearson_p,
            'Spearman Correlation': spearman_corr,
            'Spearman p-value': spearman_p
        }

    
    # Print correlation results to the console
    print("Correlation Results with User_ID:")
    for component, result in correlation_results.items():
        print(f"\nComponent: {component}")
        print(f"Pearson Correlation: {result['Pearson Correlation']} (p-value: {result['Pearson p-value']})")
        print(f"Spearman Correlation: {result['Spearman Correlation']} (p-value: {result['Spearman p-value']})")

        # Convert the correlation results into a DataFrame for easy plotting
    correlation_df = pd.DataFrame(correlation_results).T

    # Create a heatmap of the Pearson and Spearman correlations
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_df[['Pearson Correlation', 'Spearman Correlation']], annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation of Components with User_ID")
    plt.show()

    # Display a message box to inform the user that results have been printed to the console
    messagebox.showinfo("Info", "Correlation results with User_ID have been printed to the console, please check.")


"""def plot_user_component_interaction_correlation():
    global reshaped_data

    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    # Make a copy of reshaped_data to avoid modifying the original
    correlation_data = reshaped_data.copy()

    # Reset index to make User_ID a column (if it's in the index)
    if 'User_ID' in correlation_data.index.names:
        correlation_data = correlation_data.reset_index()
    
    # If User_ID is not in columns or index, we can't proceed
    if 'User_ID' not in correlation_data.columns:
        messagebox.showerror("Error", "User_ID column not found in the data.")
        return

    # Flatten columns for easier access (excluding User_ID)
    correlation_data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in correlation_data.columns]

    # Extract columns related to components (exclude 'User_ID')
    component_columns = [col for col in correlation_data.columns if col != 'User_ID']
    
    # Get user_id data
    user_ids = correlation_data['User_ID']

    # Initialize dictionary to store results
    correlation_results = {}

    for component in component_columns:
        try:
            # Convert data to numeric, replacing non-numeric values with NaN
            component_data = pd.to_numeric(correlation_data[component], errors='coerce')
            user_id_data = pd.to_numeric(user_ids, errors='coerce')
            
            # Remove any NaN values
            mask = ~(np.isnan(component_data) | np.isnan(user_id_data))
            component_data = component_data[mask]
            user_id_data = user_id_data[mask]
            
            if len(component_data) > 1:  # Need at least 2 points for correlation
                # Pearson correlation with User_ID
                pearson_corr, pearson_p = pearsonr(component_data, user_id_data)

                # Spearman correlation with User_ID
                spearman_corr, spearman_p = spearmanr(component_data, user_id_data)

                # Save correlation results
                correlation_results[component] = {
                    'Pearson Correlation': pearson_corr,
                    'Pearson p-value': pearson_p,
                    'Spearman Correlation': spearman_corr,
                    'Spearman p-value': spearman_p
                }
            else:
                print(f"Insufficient data points for correlation calculation in {component}")
                
        except Exception as e:
            print(f"Error calculating correlation for {component}: {str(e)}")
            continue

    # Print correlation results to the console
    print("\nCorrelation Results:")
    for component, result in correlation_results.items():
        print(f"\nComponent: {component}")
        print(f"Pearson Correlation: {result['Pearson Correlation']:.4f} (p-value: {result['Pearson p-value']:.4f})")
        print(f"Spearman Correlation: {result['Spearman Correlation']:.4f} (p-value: {result['Spearman p-value']:.4f})")

    # Display message box to inform the user that results have been printed to the console
    messagebox.showinfo("Info", "Correlation results have been printed to the console, please check.")"""


# Setup Tkinter GUI
root = tk.Tk()
root.title("Data Processing GUI")

# Create buttons for each step
load_clean_button = tk.Button(root, text="Load and Clean Data", command=data_load_from_csv_to_json_and_clean)
load_clean_button.pack(pady=10)

merge_button = tk.Button(root, text="Merge Data", command=merge_data)
merge_button.pack(pady=10)

reshape_button = tk.Button(root, text="Reshape Data", command=reshape_data)
reshape_button.pack(pady=10)

output_stats_button = tk.Button(root, text="Output Statistics", command=output_statistics)
output_stats_button.pack(pady=10)

plot_button_separate = tk.Button(root, text="Plot Component Interactions Separately", command=plot_component_interactions_separately)
plot_button_separate.pack(pady=10)

plot_button_heatmap = tk.Button(root, text="Plot User-Component Interactions", command=plot_user_component_correlation)
plot_button_heatmap.pack(pady=10)

root.mainloop()
