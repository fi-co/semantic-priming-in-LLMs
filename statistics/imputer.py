import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # Necessario per abilitare IterativeImputer
from sklearn.impute import IterativeImputer
from scipy.stats import t
import matplotlib.pyplot as plt
import scipy.stats as stats

# ==================== IMPUTAZIONE MULTIPLA ====================
# Carica il dataset (sostituisci 'output_results.csv' con il tuo file)
df = pd.read_csv('/Users/colomb/Desktop/INTERNSHIP_FBK/output_results.csv')
df.columns = df.columns.str.strip()

# Imposta l'imputer con sample_posterior=True per introdurre la variabilità
imputer = IterativeImputer(random_state=0, sample_posterior=True)

# Numero di imputazioni multiple
m = 5
imputed_datasets = []

for i in range(m):
    # Per variare l'imputazione, cambia il random_state per ogni ciclo
    imputer.random_state = i
    df['logprob'] = pd.to_numeric(df['logprob'], errors='coerce')
    # Seleziona le colonne numeriche
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_numeric = df[numeric_cols]
    # Imputa i dati numerici
    df_imputed_numeric = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)
    # Ora, unisci le colonne non numeriche del DataFrame originale
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    df_non_numeric = df[non_numeric_cols].reset_index(drop=True)
    df_imputed = pd.concat([df_imputed_numeric, df_non_numeric], axis=1)
    # Aggiungi dataset imputato alla lista imputed_datasets
    imputed_datasets.append(df_imputed)
    
# Ora hai 5 dataset imputati in 'imputed_datasets'

# ==================== ANALISI E POOLING DEI RISULTATI (PAIRED T-TEST) ====================
# Si assume che il dataset abbia le seguenti colonne:
#   - 'target'
#   - 'logprob'
#   - 'condition': con valori "related" e "unrelated"
#
# Poiché il file CSV originale ha la riga di intestazione, il raggruppamento parte dal primo dato reale.
# Qui ipotizziamo che ogni 2 righe (dopo la riga di intestazione) corrispondano a uno stimolo,
# in modo tale che la prima riga del gruppo sia per la condizione "related" e la seconda per "unrelated".

# Inizializziamo le liste per salvare, per ciascuna imputazione, la stima della differenza media (Q_i)
# e la varianza associata (U_i)
Q_list = []  # stime medie della differenza (related - unrelated)
U_list = []  # varianze della stima (errore standard al quadrato)

for idx, imputed_df in enumerate(imputed_datasets):
    # Reset index per sicurezza
    imputed_df = imputed_df.reset_index(drop=True)
    
    # Se per qualche motivo la prima riga contiene la stringa "target" (duplicato dell'intestazione), la rimuoviamo
    if isinstance(imputed_df.iloc[0]['target'], str) and imputed_df.iloc[0]['target'].strip().lower() == 'target':
        imputed_df = imputed_df.iloc[1:]
        imputed_df = imputed_df.reset_index(drop=True)
    
    # Crea una colonna per il gruppo: ogni 2 righe costituiscono uno stimolo
    imputed_df['group'] = imputed_df.index // 2

    # Raggruppa ogni 5 contrasti attinenti allo stesso target 
    imputed_df['target_group'] = imputed_df['group'] // 5
    
    # Raggruppa per 'target_group' e 'condition' e calcola la media del logprob per ciascun gruppo
    grouped = imputed_df.groupby(['target_group', 'condition'])['logprob'].mean().unstack()
    
    # Controlla che entrambe le condizioni siano presenti in ogni target_group
    if 'related' not in grouped.columns or 'unrelated' not in grouped.columns:
        print(f"Imputazione {idx}: alcune condizioni mancanti, saltata questa imputazione.")
        continue

    # Calcola la differenza per ogni target_group: diff = related - unrelated
    grouped['diff'] = grouped['related'] - grouped['unrelated']
    
    # Per questa imputazione:
    # Q_i è la media delle differenze per ciascun target (target_group)
    Q_i = grouped['diff'].mean()
    
    # U_i è la varianza delle differenze divisa per il numero di target_group (questo equivale
    # all'errore standard quadratico medio della stima)
    n_target_groups = len(grouped)
    U_i = grouped['diff'].var(ddof=1) / n_target_groups if n_target_groups > 1 else 0
    
    Q_list.append(Q_i)
    U_list.append(U_i)

# Se sono state ottenute stime da almeno una imputazione, procedi al pooling (paired t-test)
m_actual = len(Q_list)
if m_actual == 0:
    print("Nessuna imputazione valida per l'analisi.")
else:
    # Calcola la stima combinata Q_bar (media delle Q_i)
    Q_bar = np.mean(Q_list)
    # Stima della varianza media all'interno delle imputazioni
    U_bar = np.mean(U_list)
    # Varianza tra le imputazioni (between-imputation variance)
    B = np.var(Q_list, ddof=1)
    # Varianza totale secondo le regole di Rubin:
    T = U_bar + (1 + 1/m_actual) * B
    
    # Calcola il t-statistic
    t_stat = Q_bar / np.sqrt(T)
    
    # Calcola i gradi di libertà secondo la formula di Rubin
    if B > 0:
        df = (m_actual - 1) * (1 + U_bar / ((1 + 1/m_actual) * B))**2
    else:
        df = np.inf
    
    # Calcola il p-value (bilaterale)
    p_value = 2 * (1 - t.cdf(abs(t_stat), df))
    
    # Stampa i risultati del pooling (paired t-test)
    print("Risultati del pooling (paired t-test):")
    print(f"  Differenza media combinata (Q_bar): {Q_bar}")
    print(f"  Varianza totale combinata (T): {T}")
    print(f"  t-statistic: {t_stat}")
    print(f"  Gradi di libertà: {df}")
    print(f"  p-value: {p_value}")

# ==================== TEST WILCOXON (ALTERNATIVA) ====================
# Poiché i dati imputati non sembrano rispettare l'assunzione di normalità,
# eseguiamo un test non parametrico (Wilcoxon signed-rank) sui dati di un dataset imputato valido.
wilcoxon_performed = False

for idx, imputed_df in enumerate(imputed_datasets):
    imputed_df = imputed_df.reset_index(drop=True)
    # Rimuovi eventuali duplicati dell'intestazione
    if isinstance(imputed_df.iloc[0]['target'], str) and imputed_df.iloc[0]['target'].strip().lower() == 'target':
        imputed_df = imputed_df.iloc[1:].reset_index(drop=True)
    
    # Crea la colonna 'group': ogni 2 righe rappresentano uno stimolo (related e unrelated)
    imputed_df['group'] = imputed_df.index // 2
    # Raggruppa ogni 5 gruppi (cioè, 5 stimoli che testano lo stesso target)
    imputed_df['target_group'] = imputed_df['group'] // 5
    
    # Raggruppa per 'target_group' e 'condition' e calcola la media del logprob per ciascun gruppo
    grouped = imputed_df.groupby(['target_group', 'condition'])['logprob'].mean().unstack()
    
    # Verifica che per ogni target_group siano presenti entrambe le condizioni
    if 'related' not in grouped.columns or 'unrelated' not in grouped.columns:
        continue
    
    # Calcola la differenza per ogni target_group: diff = related - unrelated
    grouped['diff'] = grouped['related'] - grouped['unrelated']
    differences = grouped['diff'].dropna()
    
    if not differences.empty:
        from scipy.stats import wilcoxon
        # Esegui il test Wilcoxon per dati appaiati
        w_stat, w_p = wilcoxon(grouped['related'], grouped['unrelated'])
        print("Risultati del test Wilcoxon (alternativa):")
        print(f"  Test statistic: {w_stat}")
        print(f"  p-value: {w_p}")
        wilcoxon_performed = True
        break

if not wilcoxon_performed:
    print("Nessun dataset imputato valido trovato per eseguire il test Wilcoxon.")

