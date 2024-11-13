# Function to output statistics
def output_statistics():
    if reshaped_data is None:
        messagebox.showerror("Error", "Please reshape the data first.")
        return

    # Components of interest
    components_of_interest = ['Quiz', 'Lecture', 'Assignment', 'Attendence', 'Survey']
    
    # Flatten columns to access them easily
    flattened_columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in reshaped_data.columns]
    reshaped_data.colums = flattened_columns
    # Get columns that contain component-month pairs
    relevant_columns = [col for col in reshaped_data.columns if any(comp in col for comp in components_of_interest)]

    monthly_statistics = {}
    for month in ['09', '10', '11', '12']:
        month_columns = [col for col in relevant_columns if f'_{month}' in col]
        month_data = reshaped_data[month_columns]
        monthly_statistics[month] = {
            'Mean': month_data.mean().round(2),
            'Median': month_data.median(),
            'Mode': month_data.mode().iloc[0]
        }

    semester_statistics = {}
    for component in components_of_interest:
        component_columns = [col for col in relevant_columns if col.startswith(component)]
        component_data = reshaped_data[component_columns]
        semester_statistics[component] = {
            'Mean': component_data.mean().mean(),
            'Median': component_data.median().median(),
            'Mode': component_data.stack().mode().iloc[0]
        }
    
    # Prepare output message for monthly statistics
    stats_message = f"Monthly statistics:\n"
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