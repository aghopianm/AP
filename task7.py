"""#Task 7, I split this into two seperate functions, one for the initial graphs,
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