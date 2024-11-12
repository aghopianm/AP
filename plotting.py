def plot_user_component_interactions(reshaped_data):
    # Select columns representing User_ID and Component with interaction count
    # We can filter relevant data by extracting 'User_ID' and the interaction counts for each component
    
    # Extract the data by selecting columns that start with the 'User_ID' and all components
    interaction_data = reshaped_data.set_index('User_ID')

    # Plot heatmap of interactions between users and components
    plt.figure(figsize=(12, 8))
    sns.heatmap(interaction_data, annot=False, cmap='YlGnBu', fmt="d", linewidths=0.5)
    plt.title("User Interactions with Components")
    plt.xlabel("Component")
    plt.ylabel("User_ID")
    plt.show()

# Visualize component interactions
plot_component_interactions(reshaped_data)

# Calculate and plot correlation
plot_user_component_interactions(reshaped_data)