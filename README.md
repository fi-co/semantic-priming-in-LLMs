# Semantic-Priming-in-LLMs

## Applying Cognitive Psychology to LLMs: Masked Semantic Priming with GPT-4o Mini

This repository hosts the codebase, data, and documentation for an experimental study investigating semantic priming effects in large language models (LLMs)—specifically GPT-4o Mini—through a masked priming paradigm inspired by classic psycholinguistic methods.


> ⚠️ This repository is currently under construction and is subject to **ongoing updates** as the project develops. Please expect frequent changes, refinements, and additions.

---

## 🚀 Project Overview

The experiment follows a masked semantic priming paradigm, where a prime word (related or unrelated) is presented alongside a sentence with a masked target. GPT-4o is prompted to predict the missing word, and log-probabilities for the target are extracted and analyzed to detect **semantic facilitation** effects.

### How this looks like
- Integration with the **OpenAI API** for automated prompt-completion and logprob retrieval.
- A **reconstruction algorithm** to manage subword tokenization (via beam search).
- Handling of missing data through **multiple imputation** and **complete-case analysis**.
- Application of **non-parametric statistics** (e.g., Wilcoxon signed-rank test).
- Theoretical grounding in **psycholinguistics** and **cognitive modeling**.

---

## 📚 Theoretical Framework

The project draws on two major cognitive models:
- **Spreading Activation Theory** (Collins & Loftus, 1975)
- **Predictive Coding** (Friston, 2005)

These serve as frameworks to assess **whether semantic associations in LLMs reflect human-like cognition** or emerge from purely statistical regularities in training data.

---

## Status & Roadmap

This repository is a **work-in-progress**. 
I am pulling scripts from my working station to git, but it takes time to polish the code and make sure everything works well.

---

## 🧠 Author

**[Filippo Colombi]**  
Cognitive Science & NLP students  
[University of Trento | CIMeC | FBK]  

---

## 📜 License

This project is released under the [MIT License](LICENSE).  
You are free to use, modify, and distribute with attribution.
