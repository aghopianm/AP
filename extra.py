chunk_size = 1000 as an argument to the load() method 

# Merge activity_log and component_codes on Component
    #I am going to merge this one without merging in chunks, however for the second merge
    #I will have to merge in chunks as memory was running out.
    """merged_data = pd.merge(data1, data2, on="Component", how="outer")
    print(merged_data.head())
    merged_data = pd.merge(data3, merged_data, on="User Full Name *Anonymized", how="outer")
    print(merged_data.head())"""
    """first_10_rows = merged_data.head(10)
    first_10_rows.to_json("first10rows.json", orient='records', lines=False, indent=4)
"""

    # Save the merged data into a new CSV
    #merged_data.to_csv("merged_data.csv", index=False, encoding='utf-8')
    # merged_data.to_json("data_initial_join.json", orient='records', lines=True)

    # Convert the merged data to JSON (use the 'records' format with indentation)
    """merged_data = merged_data.to_dict(orient='records')
    merged_data.to_json("data_saved.json", orient='records', lines=True, indent=4)"""



    """with open("data_cleaned.json", "w") as json_file:
        json_file.write("[")  # Start JSON array

        # Track if this is the first chunk to handle commas properly in JSON
        first_chunk = True

        # Process USER_LOG in chunks and write each merged chunk to JSON
        for i, user_chunk in enumerate(pd.read_csv(file_path["USER_LOG"], chunksize=chunk_size)):
            user_chunk.columns = user_chunk.columns.str.strip()
            
            # Merge each chunk with merged_data
            chunk_merged = pd.merge(user_chunk, merged_data, on="User Full Name *Anonymized", how="outer")

            # Write the chunk to the JSON file in the correct format
            if not first_chunk:
                json_file.write(",")  # Add comma before each new chunk, except the first
            chunk_merged.to_json(json_file, orient='records', lines=True)

            first_chunk = False
            print(f"Processed and saved chunk {i + 1}")

        json_file.write("]")  # Close JSON array

    print("Data saved to data_cleaned.json")

    return chunk_merged"""
    return toilet

file_path = {
    "ACTIVITY_LOG": "ACTIVITY_LOG.csv",
    "COMPONENT_CODES": "COMPONENT_CODES.csv",
    "USER_LOG": "USER_LOG.csv"
}