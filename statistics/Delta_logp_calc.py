import pandas as pd

# Load the synthetic dataset
data_path = "/Users/colomb/Desktop/INTERNSHIP_FBK/output_results.csv"
data = pd.read_csv(data_path)

# Initialize a list to store the processed data
result = []

# Group data by target to compute deltas
grouped = data.groupby("target")

for target, group in grouped:
    # Ensure the group has exactly two rows (one for each condition)
    if len(group) == 2:
        # Extract logprobs for related and unrelated conditions
        related_logprob = group.loc[group["condition"] == "related", "logprob"].values[0]
        unrelated_logprob = group.loc[group["condition"] == "unrelated", "logprob"].values[0]

        # Compute Delta LogP
        delta_logp = related_logprob - unrelated_logprob

        # Append the result to the processed data
        result.append({
            "target": target,
            "Delta_LogP": delta_logp,
            "related_logprob": related_logprob,
            "unrelated_logprob": unrelated_logprob
        })



# Convert processed data to a new DataFrame
processed_df = pd.DataFrame(result)
# Sort the DataFrame by the target column to ensure proper order
processed_df = processed_df.sort_values(by="target", key=lambda col: col.str.extract(r'(\d+)$').astype(int)[0]).reset_index(drop=True)

# Calculate mean and standard deviation of Delta_LogP
delta_mean = processed_df["Delta_LogP"].mean()
delta_std = processed_df["Delta_LogP"].std()

# Add a summary row with mean and standard deviation
summary_row = {
    "target": "summary",
    "Delta_LogP": f"mean: {delta_mean:.4f}, std: {delta_std:.4f}",
    "related_logprob": None,
    "unrelated_logprob": None
}
processed_df = pd.concat([processed_df, pd.DataFrame([summary_row])], ignore_index=True)

# Save the processed data to a new CSV file
output_path = "SYNTHETIC/delta_results.csv"
processed_df.to_csv(output_path, index=False)

print(f"Processed dataset with deltas has been saved as {output_path}.")
