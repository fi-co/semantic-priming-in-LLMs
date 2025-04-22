import pandas as pd

# Load the CSV
file_path = "full_stimuli.csv"
df = pd.read_csv(file_path, sep=";")

# Filtra le righe con un certo criterio
filtered_df = df[df["ID"].str.match(r"^\d*[02468]_prime$|^\d*[02468]_unrelated$")]

# Salva il nuovo CSV
output_path = "stimuli_experiment_1.csv"
filtered_df.to_csv(output_path, index=False)

print(f"Filtering complete. Saved in {output_path}")