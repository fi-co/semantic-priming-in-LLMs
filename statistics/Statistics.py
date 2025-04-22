import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_rel, skew, wilcoxon
import matplotlib.pyplot as plt


#               ===============  STATISTICAL TEST  ===============
#
# Operative Question: Does the condition (related vs unrelated prime) significantly 
# influence the logprobabilities of target words?

# H0: the mean difference in logprobs (ΔLogP) between related and unrelated conditions
# is zero: μ(ΔLogP)​=0

# H1: the mean difference is not zero: μ(ΔLogP)​≠0


# Prepare the data
df = pd.read_csv('output_results.csv')
df.columns = df.columns.str.strip() # Remove spaces

# Ensure column 'lgoprob' is numeric
df['logprob'] = pd.to_numeric(df['logprob'], errors = 'coerce')

# Suppose each couple of rows represents the two conditions for the same target 
df = df.reset_index(drop=True)
df['group'] = df.index // 2

# Filter complete couplets
def valid_pair (group_df):
    return (group_df.shape[0] == 2) and group_df['logprob'].notnull().all()

# Group and filter
valid_pairs = df.groupby('group').filter(valid_pair)
pairs_pivot = valid_pairs.pivot(index='group', columns="condition", values="logprob")

# Check if each gorup has both conditions
if 'related' in pairs_pivot.columns and 'unrelated' in pairs_pivot.columns:
    # Compute differential for each couplet
    pairs_pivot['delta'] = pairs_pivot['related'] - pairs_pivot['unrelated']
    
    # T-test 
    t_stat, p_value = ttest_rel(pairs_pivot['related'], pairs_pivot['unrelated'])

    print("Paired-test T-stat:")
    print("Number of couplets:", pairs_pivot.shape[0])
    print(f"t-stat: {t_stat}")
    print(f"p-value: {p_value}")

#           ============= EFFECT SIZE (Cohen's d) ===============
#
# Supponendo di avere la colonna 'delta' nel DataFrame pairs_pivot (ottenuto nel paired t-test)
differences = pairs_pivot['delta'].dropna()  # Assicuriamoci di escludere eventuali NaN

# Calcola la media e la deviazione standard delle differenze
mean_delta = differences.mean()
std_delta = differences.std(ddof=1)

# Calcola Cohen's d per campioni appaiati:
cohens_d = mean_delta / std_delta

print("Effetto size (Cohen's d):", cohens_d)

# Calcola e stampa la skewness
diff_skew = skew(pairs_pivot['delta'])
print("Skewness delle differenze:", diff_skew)

# Ora, crea l'istogramma delle differenze
plt.figure(figsize=(8, 4))
plt.hist(pairs_pivot['delta'], bins=20, edgecolor='k', alpha=0.7)
plt.axvline(0, color='red', linestyle='dashed', linewidth=2)  # linea a zero per riferimento
plt.title("Istogramma delle Differenze (related - unrelated)")
plt.xlabel("Differenza")
plt.ylabel("Frequenza")
plt.show()

"""
#           ===============  NORMALITY TEST (Q-Q plot)  ===============
#
# Esecuzione del test di Shapiro-Wilk per la normalità
shapiro_stat, shapiro_p = stats.shapiro(differences)
print("Shapiro-Wilk Test:")
print("Statistic:", shapiro_stat)
print("p-value:", shapiro_p)
# Creazione del Q-Q plot per le differenze
plt.figure(figsize=(6, 6))
stats.probplot(differences, dist="norm", plot=plt)
plt.title("Q-Q Plot delle Differenze (related - unrelated)")
plt.xlabel("Quantili teorici")
plt.ylabel("Quantili osservati")
plt.show()

#           ===============  DIFFERENCES DISTRIBUTION  ===============
#
# Creazione dell'istogramma
plt.figure(figsize=(8, 4))
plt.hist(differences, bins=20, edgecolor='k', alpha=0.7)
plt.title("Istogramma delle Differenze")
plt.xlabel("Differenza (related - unrelated)")
plt.ylabel("Frequenza")
plt.show()

# Creazione del boxplot
plt.figure(figsize=(4, 6))
plt.boxplot(differences, vert=True, patch_artist=True)
plt.title("Boxplot delle Differenze")
plt.ylabel("Differenza (related - unrelated)")
plt.show()
"""
# Esecuzione del test di Shapiro-Wilk per la normalità
shapiro_stat, shapiro_p = stats.shapiro(differences)
print("Shapiro-Wilk Test:")
print("Statistic:", shapiro_stat)
print("p-value:", shapiro_p)

#           ===============  WILCOXON SIGNED-RANK TEST  ===============
import pandas as pd

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
