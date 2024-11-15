import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy.stats import pearsonr, spearmanr

sns.set(style="whitegrid")

#Main variables that are used through the programme are initialised here at the top. 
#I have had to do this due to scope issues. 
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

    #Load all CSVs to a pandas dataframe, pandas > np in this case as we are going to have to
    #heavily manipulate the dataset and reshape it.
    activity_logs_dataframe = pd.read_csv(file_paths[0])
    component_codes_dataframe = pd.read_csv(file_paths[1])
    user_log_dataframe = pd.read_csv(file_paths[2])

    #Strip whitespace
    activity_logs_dataframe.columns = activity_logs_dataframe.columns.str.strip()
    component_codes_dataframe.columns = component_codes_dataframe.columns.str.strip()
    user_log_dataframe.columns = user_log_dataframe.columns.str.strip()

    #Clean activity logs first
    activity_logs_dataframe.fillna({'Component': 'Unknown', 'Action': 'Unknown', 'Target': 'Unknown', 'User Full Name *Anonymized': 0}, inplace=True)
    activity_logs_dataframe['Target'] = activity_logs_dataframe['Target'].str.replace('_', '', regex=False)
    activity_logs_dataframe['Component'] = activity_logs_dataframe['Component'].str.replace('_', '', regex=False)
    
    #Clean user logs next
    user_log_dataframe['Date'] = pd.to_datetime(user_log_dataframe['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
    user_log_dataframe['Date'] = user_log_dataframe['Date'].dt.strftime('%d/%m/%Y')
    user_log_dataframe['Date'].fillna('25/11/2023', inplace=True)
    user_log_dataframe['Time'] = user_log_dataframe['Time'].str.strip()
    user_log_dataframe['Time'].fillna('00:00:00', inplace=True)

    #Clean component codes last
    component_codes_dataframe['Component'] = component_codes_dataframe['Component'].str.replace('_', '', regex=False)
    component_codes_dataframe['Code'] = component_codes_dataframe['Code'].str.replace('_', '', regex=False)

    messagebox.showinfo("Info", "Data loaded and cleaned successfully.")

#This function saves the three cleaned dataframes to three seperate JSON files.
def save_cleaned_data():
    global activity_logs_dataframe, component_codes_dataframe, user_log_dataframe
    if activity_logs_dataframe is None or component_codes_dataframe is None or user_log_dataframe is None:
        messagebox.showerror("Error", "Data not loaded. Please load and clean data first.")
        return

    activity_logs_dataframe.to_json("activity_logs_cleaned.json", orient="records", lines=False, indent=4)
    component_codes_dataframe.to_json("component_codes_cleaned.json", orient="records", lines=False, indent=4)
    user_log_dataframe.to_json("user_log_cleaned.json", orient="records", lines=False, indent=4)
    messagebox.showinfo("Info", "Cleaned data saved as JSON files.")

#Data manipulation and outputs tasks:
#Task 1 and 2
def remove_and_rename():
    global activity_logs_dataframe, user_log_dataframe, component_codes_dataframe
    if activity_logs_dataframe is None or component_codes_dataframe is None or user_log_dataframe is None:
        messagebox.showerror("Error", "Please load and clean the data first.")
        return
    #Remove task:
    # Filter out "System" and "Folder" components
    activity_logs_dataframe = activity_logs_dataframe[~activity_logs_dataframe['Component'].isin(['System', 'Folder'])]
    #Rename task:
    activity_logs_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)
    user_log_dataframe.rename(columns={'User Full Name *Anonymized': 'User_ID'}, inplace=True)

    messagebox.showinfo("Info", "Thank you, you have removed and renamed successsfully")

#Task 3 
# Function to merge data
def merge_data():
    global fully_merged_dataset
    if activity_logs_dataframe is None or component_codes_dataframe is None or user_log_dataframe is None:
        messagebox.showerror("Error", "Please load and clean the data first, and remove&rename.")
        return
    messagebox.showinfo("Info", "This process may take up to 5-10 minutes. Please wait.")
    first_merge = pd.merge(activity_logs_dataframe, component_codes_dataframe, on='Component', how='left')
    fully_merged_dataset = pd.merge(first_merge, user_log_dataframe, on='User_ID', how='left')
    messagebox.showinfo("Info", "Data merged successfully.")

#Task 4
# Function to reshape data
def reshape_data():
    global reshaped_data
    if fully_merged_dataset is None:
        messagebox.showerror("Error", "Please merge the data first.")
        return
    messagebox.showinfo("Info", "This process may take up to 5-10 minutes. Please wait.")
    
    #I am string slicing here to get the month.
    fully_merged_dataset['Month'] = fully_merged_dataset['Date'].str[3:5]
    #Task 5 of counting, interaction count = task 5
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

def save_large_json(dataframe, filename, chunk_size=1000):
    #json files opened for writing.
    with open(filename, 'w') as f:
        #initiate array.
        f.write("[\n")
        
        #dataframe looped through in chunks ( size can be given by me here)
        for i in range(0, len(dataframe), chunk_size):
            #get chunk
            chunk = dataframe.iloc[i:i + chunk_size]
            #pass to a json
            chunk_json = chunk.to_json(orient="records", lines=False)
            #then take the json to a file.
            f.write(chunk_json[1:-1])  # Exclude the outer brackets for each chunk
            
            # addcomma between chunks. except after the last chunk to avoid erros.
            if i + chunk_size < len(dataframe):
                f.write(",\n")
            #progress printed to console so I dont think its not working.
            print(f"Printed chunk {i // chunk_size + 1} for {filename}")

        #close array. 
        f.write("\n]")

#purpose of this function is a helper function specifically for the reshaped data as it was throwing
#errors when I used savelargejson() on the reshaped data..
def save_reshaped_data():
    global reshaped_data
    if reshaped_data is None:
        messagebox.showerror("Error", "Data not yet reshaped.")
        return
    
    #reshaped data was throwing erros when using larger chunk sizes so I have put a smaller size. 
    save_large_json(reshaped_data, "reshaped_data.json", chunk_size=10)
    print("Reshaped data saved successfully.")

#Function to save the fully merged and reshaped data to seperate json files for future use and to
#maintain the programme moving forward.
def save_prepared_data():
    global fully_merged_dataset, reshaped_data
    if fully_merged_dataset is None or reshaped_data is None:
        messagebox.showerror("Error", "Data not yet merged and reshaped.")
        return
    messagebox.showinfo("Info", "This process will take a VERY LONG TIME - Please wait up to 20 mins")
    save_large_json(fully_merged_dataset, "fully_merged_dataset.json")
    print("Complete, now I will do the reshaped data.")
    save_reshaped_data()
    messagebox.showinfo("Info", "Prepared data saved as JSON files.")

#Function to load the fully merged and reshaped data from seperate files so you don't have to 
#go through the whole process of cleaning, merging, loading etc from the csv every time if you have
#a prepared dataset.

def load_large_json(filename, chunk_size=1000):

    #generator function within the load large json. this handles chunk generation 
    def json_chunk_generator(file_obj, chunk_size):
        buffer = []
        in_array = False
        for line in file_obj:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle start of array
            if line == '[':
                in_array = True
                continue
            # Handle end of array
            elif line == ']':
                if buffer:
                    yield '[' + ','.join(buffer) + ']'
                break
                
            # Handle regular lines
            if in_array:
                # Remove trailing comma if present
                if line.endswith(','):
                    line = line[:-1]
                buffer.append(line)
                
                if len(buffer) >= chunk_size:
                    yield '[' + ','.join(buffer) + ']'
                    buffer = []
    
    data_chunks = []
    try:
        with open(filename, 'r') as f:
            for chunk_json in json_chunk_generator(f, chunk_size):
                try:
                    chunk_df = pd.read_json(chunk_json, orient='records')
                    data_chunks.append(chunk_df)
                    print(f"Loaded chunk with {len(chunk_df)} records from {filename}")
                except ValueError as e:
                    print(f"Error processing chunk: {e}")
                    continue
                    
        if not data_chunks:
            raise ValueError("No valid data chunks were loaded")
            
        result = pd.concat(data_chunks, ignore_index=True)
        #I will get 50,000 at a time from my pre existing JSON, this takes a very long time on my
        #m1 macbook air and I am not sure if this is the best way to programme this.
        #perhaps a solution moving forward could be using threads for IO operaitons, there is
        #lots of issues in this programme with IO that could be optimised through threading.
        print(f"Successfully loaded total of {len(result)} records from {filename}")
        return result
        
    except Exception as e:
        print(f"Error loading file {filename}: {str(e)}")
        raise

def load_prepared_data():
    global fully_merged_dataset, reshaped_data
    try:
        messagebox.showinfo("Info", "This process will take a VERY LONG TIME - Please wait up to 60 mins depending on your hardware")
        
        #Load the fully merged dataset with a larger chunk size
        print("Loading fully merged dataset...")
        fully_merged_dataset = load_large_json("fully_merged_dataset.json", chunk_size=500)
        
        #Load the reshaped data with a smaller chunk size since it is much, much wider.
        print("Loading reshaped data...")
        reshaped_data = load_large_json("reshaped_data.json", chunk_size=100)
        
        messagebox.showinfo("Info", "Prepared data loaded successfully.")
        
    except Exception as e:
        messagebox.showerror("Error", f"Error loading prepared data: {str(e)}")
        print(f"Detailed error: {str(e)}")

#Task 6
def output_statistics():
    global reshaped_data
    
    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    #throughout my programme I will keep making copyies to avoid messing with the global reshaped data
    #variable as I had issues when flattening columns from one function to another.
    manipulated_data = reshaped_data.copy()

    # Components of interest that I need to extract.
    components_of_interest = ['Quiz', 'Lecture', 'Assignment', 'Attendance', 'Survey']
    
    # Flatten columns to access them easily in the copied data.
    manipulated_data.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in manipulated_data.columns]

    # Get columns that contain component-month pairs
    relevant_columns = [col for col in manipulated_data.columns if any(comp in col for comp in components_of_interest)]

    #Task 6a
    # Monthly statistics (September, October, November, December)
    monthly_statistics = {}
    for month in ['09', '10', '11', '12']:
        month_columns = [col for col in relevant_columns if f'_{month}' in col]
        month_data = manipulated_data[month_columns]

        monthly_statistics[month] = {}
        
        #mean for each month
        monthly_statistics[month]['Mean'] = month_data.mean().round(2)
        
        #median for each month
        monthly_statistics[month]['Median'] = month_data.median().to_dict()
        
        #first mode returned for each month with [0]
        mode_values = month_data.mode()
        if not mode_values.empty:
            monthly_statistics[month]['Mode'] = mode_values.iloc[0].to_dict()  #each mode for component.
        else:
            #Handle the case where there's no mode as i think thats more useful than returning
            #nan or 0 to the console.
            monthly_statistics[month]['Mode'] = 'No mode'  
    #Task 6b
    #Semester statistics 09-12, september to december.
    semester_statistics = {}
    for component in components_of_interest:
        # get components across every single month
        component_columns = [col for col in relevant_columns if col.startswith(component)]
        
        component_data = manipulated_data[component_columns]
        
        #all month means calculated.
        semester_statistics[component] = {}
        semester_statistics[component]['Mean'] = component_data.mean().mean()  # Mean of all months combined
        
        #medians calculated across all months. (i.e: i am getting the median of medians, mean of means etc)
        semester_statistics[component]['Median'] = component_data.median().median()  #Median of all months combined
        
        #all  month modes.
        # Combine data from all months for the component and calculate mode
        # Stack all the month data into a single column, I did it this way as I did not want to have many
        #modes simply for the semester, it didnt really make sense having individual month modes for the
        #semester as this was what I was getting when I coded initially.
        combined_data = component_data.stack() 
        mode_values = combined_data.mode()
        if not mode_values.empty:
            semester_statistics[component]['Mode'] = mode_values.iloc[0] 
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

    #keep me updated.
    messagebox.showinfo("Info", "Statistics have been printed to the console, please check.")

#Task 7 - the first part of the task, plotting the graphs
def plot_bar_graphs():
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

#Task 7 seond half, calculating and plotting correlation
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

    
    #correlation results printed to menu, optional I could have saved this to a file for future readability
    print("Correlation Results with User_ID:")
    for component, result in correlation_results.items():
        print(f"\nComponent: {component}")
        print(f"Pearson Correlation: {result['Pearson Correlation']} (p-value: {result['Pearson p-value']})")
        print(f"Spearman Correlation: {result['Spearman Correlation']} (p-value: {result['Spearman p-value']})")

        #i will plot using a dataframe so results must be converted.
    correlation_df = pd.DataFrame(correlation_results).T

    #heatmap for both pearson and spearman correlation.
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_df[['Pearson Correlation', 'Spearman Correlation']], annot=False, cmap='coolwarm', center=0)
    plt.title("Correlation of Components with User_ID")
    plt.show()

    messagebox.showinfo("Info", "Correlation results with User_ID have been printed to the console, please check.")

#POTENTIAL FINAL SOLUTION FOR GUI. 
# Setup Tkinter GUI
root = tk.Tk()
root.title("GUI for my prototype application")

#Main menu bar
menubar = tk.Menu(root)
root.config(menu=menubar)

"""#OPTIONAL STYLING
style = ttk.Style()
style.configure(
    "TLabel",  # Default style for all ttk.Label widgets
    background="#e6f3ff",
    foreground="#003366",
    font=("Arial", 12),
    padding=10
)"""

#All things data processing in one menu
data_processing_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Data Preprocessing", menu=data_processing_menu)
data_processing_menu.add_command(label="Load and Clean Data", command=data_load_from_csv_to_json_and_clean)
data_processing_menu.add_command(label="Remove and Rename", command=remove_and_rename)
data_processing_menu.add_command(label="Merge Data", command=merge_data)
data_processing_menu.add_command(label="Reshape Data", command=reshape_data)

#data storage menu
data_storage_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Data Storage and Loading", menu=data_storage_menu)
data_storage_menu.add_command(label="Save Cleaned Data to JSON (individual files)", command=save_cleaned_data)
data_storage_menu.add_command(label="Save Prepared Data to JSON (one big JSON file)", command=save_prepared_data)
data_storage_menu.add_command(label="Load Prepared Data from JSON (merged and reshaped)", command=load_prepared_data)

#statistics and visualiation menu
visualization_menu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="Statistics & Visualization", menu=visualization_menu)
visualization_menu.add_command(label="Output Statistics", command=output_statistics)
visualization_menu.add_command(label="Component Interactions Bar Graphs", command=plot_bar_graphs)
visualization_menu.add_command(label="User-Component Correlation Heatmap/Correlation", command=plot_user_component_correlation)

#status bar at the bottom, not sure how much value this adds but it looks good.
status_bar = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
status_bar.pack(side=tk.BOTTOM, fill=tk.X)

#instruction and welcome menu
welcome_label = ttk.Label(root, text="Prototype application, welcome!\n\nPlease use the menus above to:\n1. Process your data\n2. Save/Load data\n3. Visualize results and output statistics", justify=tk.CENTER, padding=20)
welcome_label.pack(expand=True)

root.mainloop()

#POTENTIALLY USE THIS GUI INSTEAD
"""# Setup Tkinter GUI
root = tk.Tk()
root.title("Data Processing GUI")
# Create buttons for each step
load_clean_button = tk.Button(root, text="Load and Clean Data", command=data_load_from_csv_to_json_and_clean)
load_clean_button.pack(pady=10)

save_cleaned_data_button = tk.Button(root, text="Save cleaned CSV data to JSON", command=save_cleaned_data)
save_cleaned_data_button.pack(pady=10)

# Create buttons for each step
remove_and_rename_button = tk.Button(root, text="Remove and rename", command=remove_and_rename)
remove_and_rename_button.pack(pady=10)

merge_button = tk.Button(root, text="Merge Data", command=merge_data)
merge_button.pack(pady=10)

reshape_button = tk.Button(root, text="Reshape Data", command=reshape_data)
reshape_button.pack(pady=10)

# Create buttons for each step
save_prepared_data_button = tk.Button(root, text="save cleaned and prepared data to JSON", 
                                      command=save_prepared_data)
save_prepared_data_button.pack(pady=10)

# Create buttons for each step
load_JSON_prepared_data_button = tk.Button(root, text="Load both merged data and reshaped data from JSOn", 
                                           command=load_prepared_data)
load_JSON_prepared_data_button.pack(pady=10)

output_stats_button = tk.Button(root, text="Output Statistics", command=output_statistics)
output_stats_button.pack(pady=10)

plot_button_separate = tk.Button(root, text="Plot Component Interactions Separately", command=plot_bar_graphs)
plot_button_separate.pack(pady=10)

plot_button_heatmap = tk.Button(root, text="Plot User-Component Interactions", command=plot_user_component_correlation)
plot_button_heatmap.pack(pady=10)

root.mainloop()"""