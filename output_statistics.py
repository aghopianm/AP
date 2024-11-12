"""def output_statistics(reshaped_data):
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
        
        # Mode for this month (returning all modes per component)
        monthly_statistics[month]['Mode'] = month_data[month_columns].mode().T.apply(lambda x: x.tolist(), axis=1)

    # Print out the statistics per month
    print("Statistics per month:")
    for month, stats in monthly_statistics.items():
        print(f"\nMonth: {month}")
        print("Mean:\n", stats['Mean'])
        print("Median:\n", stats['Median'])
        print("Mode:\n", stats['Mode'])

    # Calculate overall semester statistics (September to December)
    semester_columns = [col for col in relevant_columns if col[1] in ['09', '10', '11', '12']]
    semester_data = reshaped_data.loc[:, [('User_ID', '')] + semester_columns]
    
    semester_statistics = {}
    semester_statistics['Mean'] = semester_data[semester_columns].mean()
    semester_statistics['Median'] = semester_data[semester_columns].median()
    semester_statistics['Mode'] = semester_data[semester_columns].mode().T.apply(lambda x: x.tolist(), axis=1)

    # Print out the overall semester statistics
    print("\nStatistics for the entire 13-week semester (September to December):")
    print("Mean:\n", semester_statistics['Mean'])
    print("Median:\n", semester_statistics['Median'])
    print("Mode:\n", semester_statistics['Mode'])
    
    return monthly_statistics, semester_statistics"""

"""def output_statistics(reshaped_data):
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
    semester_columns = [col for col in relevant_columns if col[1] in ['09', '10', '11', '12']]
    semester_data = reshaped_data.loc[:, [('User_ID', '')] + semester_columns]
    
    semester_statistics = {}
    
    # Mean for the semester (average of monthly means)
    semester_statistics['Mean'] = semester_data[semester_columns].mean().mean()  # Mean of means
    
    # Median for the semester (median of the monthly medians)
    semester_statistics['Median'] = semester_data[semester_columns].median().median()  # Median of medians
    
    # Mode for the semester (first mode across all months)
    # We'll extract the first mode for each component from each month
    modes = []
    for month in ['09', '10', '11', '12']:
        month_columns = [col for col in relevant_columns if col[1] == month]
        month_data = reshaped_data.loc[:, [('User_ID', '')] + month_columns]
        month_mode = month_data[month_columns].mode().iloc[0]  # First mode for the month
        modes.append(month_mode)
    
    # For semester, we get the mode that is most frequent across the months
    semester_statistics['Mode'] = pd.concat(modes, axis=0).mode().iloc[0]  # Mode of the modes

    # Print out the overall semester statistics
    print("\nStatistics for the entire 13-week semester (September to December):")
    print("Mean:\n", semester_statistics['Mean'])
    print("Median:\n", semester_statistics['Median'])
    print("Mode:\n", semester_statistics['Mode'])
    
    return monthly_statistics, semester_statistics"""