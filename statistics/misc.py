import pandas as pd
from scipy.stats import wilcoxon
"""
# Carica il file CSV e pulisci i nomi delle colonne
df = pd.read_csv('/Users/colomb/Desktop/INTERNSHIP_FBK/output_results.csv')
df.columns = df.columns.str.strip()

# Converte la colonna 'logprob' da string a float, forzando eventuali errori a NaN
df['logprob'] = pd.to_numeric(df['logprob'], errors='coerce')

# Reset dell'indice per assicurarsi che i dati siano ordinati correttamente
df = df.reset_index(drop=True)

# Assegna un numero di gruppo: ogni coppia di righe (2 righe consecutive) rappresenta uno stimolo,
# dove la prima riga è per "related" e la seconda per "unrelated"
df['group'] = df.index // 2

# Filtra solo le coppie in cui entrambe le condizioni hanno un valore valido in 'logprob'
def valid_pair(group):
    return (group.shape[0] == 2) and group['logprob'].notnull().all()

valid_df = df.groupby('group').filter(valid_pair)

# Riorganizza i dati in modo da avere le condizioni in colonne
pairs = valid_df.pivot(index='group', columns='condition', values='logprob')

# Verifica che entrambi i campi siano presenti
if 'related' in pairs.columns and 'unrelated' in pairs.columns:
    # Esegui il test di Wilcoxon per dati appaiati
    stat, p_value = wilcoxon(pairs['related'], pairs['unrelated'])
    print("Wilcoxon test statistic:", stat)
    print("p-value:", p_value)
else:
    print("Non sono presenti entrambe le condizioni 'related' e 'unrelated' in ogni coppia.")
"""

df = pd.read_csv('output_results.csv')

column_name = 'logprob                 '

data = pd.to_numeric(df[column_name].iloc[1:], errors='coerce').dropna()

summary_logprob = data.describe()
print(summary_logprob)