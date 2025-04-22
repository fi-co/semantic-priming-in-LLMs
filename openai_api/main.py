import csv
import pandas as pd
import re
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load API key
load_dotenv(dotenv_path="openai_api/.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API Key is missing. Please set it in the .env file.")

client = OpenAI(api_key=api_key)


def run_experiment(file_path, output_file="output_results.csv", logprobs_file="logprobs_analysis.csv", model="gpt-4o-mini", shuffle=False):
    """
    Processa le frasi da un file CSV, le invia a GPT e valuta i logprobs.

    Vengono salvati:
    - `output_results.csv`: File con 3 colonne: target | logprob | condition.
      - target: viene generato un pattern (word_0, word_0, word_2, word_2, ...)
      - logprob: valore combinato ottenuto dall'interazione con GPT.
      - condition: alterna "related" e "unrelated".
    - `logprobs_analysis.csv`: Informazioni dettagliate sui logprobs per analisi qualitative.
    """
    try:
        data = pd.read_csv(file_path)
        if shuffle:
            data = data.sample(frac=1).reset_index(drop=True)

        results = []
        logprobs_data = []
        i = 0

        for _, row in data.iterrows():
            original_target = row["Target"]
            sentence = row["Sentence"]
            prime_word = row["Word"]
            input_text = f"[{prime_word}]. {sentence}"

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You do text-completion. I will provide you a sentence with a blank '...', your task is to return a single word to complete it."},
                        {"role": "user", "content": input_text}
                    ],
                    temperature=0,
                    max_completion_tokens=8,
                    logprobs=True,
                    top_logprobs=15
                )

                if not hasattr(response.choices[0].logprobs, 'content'):
                    raise ValueError("Struttura dei logprobs mancante o non valida.")

                logprobs = response.choices[0].logprobs.content
                reconstructed_target, combined_logprob = reconstruct_target(original_target, logprobs)

                logprobs_data.append({
                    "input_text": input_text,
                    "target": original_target,
                    "logprobs_used": [(lp.token, lp.logprob) for lp in logprobs],
                    "reconstructed_target": reconstructed_target,
                    "success": reconstructed_target is not None
                })

            except Exception as e:
                combined_logprob = None

            target_pattern = f"word_{(i // 2) * 2}"
            condition = "related" if i % 2 == 0 else "unrelated"
            results.append({
                "target": target_pattern,
                "logprob": combined_logprob,
                "condition": condition
            })
            i += 1

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)

        logprobs_df = pd.DataFrame(logprobs_data)
        logprobs_df.to_csv(logprobs_file, index=False, quoting=csv.QUOTE_ALL)

        return f"Processing complete. Results saved to '{output_file}', logprobs saved to '{logprobs_file}'."

    except Exception as e:
        return f"An error occurred: {e}"


# Helper functions

def reconstruct_target(target, logprobs):
    """
    Ricostruisce il target a partire dai sub-token presenti in logprobs,
    utilizzando un approccio di beam search per bilanciare l'estensione
    (cioè, il match con il target) e la somma dei logprob (che rappresenta la probabilità).
    
    L'algoritmo:
      - Mantiene una "beam" (lista) di candidate state, ciascuno rappresentato come
        (reconstructed, aggregated_logprob).
      - Per ogni token nei logprobs, prova a estendere ogni candidato usando sia
        il token principale che le sue alternative.
      - Se l'estensione mantiene il prefisso valido rispetto al target (cioè, 
        target_norm.startswith(new_str) è True), il candidato viene mantenuto.
      - Al termine, tra tutti i candidati che hanno ricostruito esattamente il target,
        si sceglie quello con il miglior (cioè, maggiore) aggregated_logprob.
      
    Se non viene ricostruito il target completo, restituisce (None, None).
    """
    target_norm = normalize(target)
    # Inizialmente la beam contiene lo stato vuoto, con logprob zero.
    beam = [("", 0.0)]
    
    for token_logprob in logprobs:
        new_beam = []
        # Raccogliamo i candidati del token corrente:
        candidates = []
        main_token = normalize(token_logprob.token)
        candidates.append((main_token, token_logprob.logprob))
        for alt in token_logprob.top_logprobs:
            alt_token = normalize(alt.token)
            candidates.append((alt_token, alt.logprob))
        
        # Per ogni candidato attuale, tentiamo di estenderlo con ciascun candidato del token
        for (reconstructed, agg_logprob) in beam:
            for candidate_token, candidate_logprob in candidates:
                new_str = reconstructed + candidate_token
                # Verifica: la nuova stringa deve essere un prefisso valido del target
                if target_norm.startswith(new_str):
                    new_beam.append((new_str, agg_logprob + candidate_logprob))
        # Inoltre, se qualche candidato nella beam è già completo, lo portiamo avanti
        for (reconstructed, agg_logprob) in beam:
            if reconstructed == target_norm:
                new_beam.append((reconstructed, agg_logprob))
        beam = new_beam
        if not beam:
            # Se in qualche iterazione nessun candidato può essere esteso, esci dal ciclo.
            break

    # Alla fine, scegli tra i candidati che hanno ricostruito esattamente il target,
    # quello con aggregated_logprob maggiore (cioè, meno negativo).
    full_candidates = [(r, lp) for (r, lp) in beam if r == target_norm]
    if full_candidates:
        best_candidate = max(full_candidates, key=lambda x: x[1])
        return best_candidate
    else:
        return (None, None)
    

def normalize(token):
    """
    Normalizza il token:
      - Converte in minuscolo.
      - Rimuove spazi e caratteri speciali ai bordi.
      - Rimuove una 's' finale per gestire in modo semplice i plurali.
    """
    token = token.lower().strip()
    token = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', token)
    token = re.sub(r's$', '', token)
    return token


# Esempio di utilizzo
run_experiment("/Users/colomb/Desktop/INTERNSHIP_FBK/stimuli_experiment_1.csv", shuffle=False)
print("Processing complete. Results saved to 'output_results.csv', logprobs saved to 'logprobs_analysis.csv'.")