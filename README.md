# Fine-Tuning a LLaMA 2 Model on Medical Text Data Using Hugging Face

This notebook demonstrates how to **fine-tune a pre-trained LLaMA 2 model** using **Hugging Face Transformers** and **Google Colab**.  
The model used is [`aboonaji/llama2finetune-v2`](https://huggingface.co/aboonaji/llama2finetune-v2), and the fine-tuning dataset is a **medical text dataset** also hosted on Hugging Face.

---

##  Overview

| Step | Description |
|------|--------------|
| **1. Installing and Importing Libraries** | Installed and imported all required packages like `torch`, `transformers`, `trl`, `peft`, `datasets`, and `bitsandbytes` directly in Colab. |
| **2. Loading the Base Model** | Loaded the pre-trained LLaMA 2 model (`aboonaji/llama2finetune-v2`) from Hugging Face using quantization (4-bit) to optimize GPU memory. |
| **3. Tokenizer Setup** | Loaded the tokenizer from the same Hugging Face model, set the padding side to *right*, and used the end-of-sequence token as the padding token. |
| **4. Setting Training Arguments** | Defined `TrainingArguments` such as batch size, gradient accumulation, and max steps to control the fine-tuning process. |
| **5. Supervised Fine-Tuning (SFT)** | Created an **SFT Trainer** (supervised fine-tuning trainer) using the medical dataset. The model learns medical domain terminology and relationships from this dataset. |
| **6. Training Execution** | Ran the training process in Colab GPU environment, ensuring GPU memory optimization by adjusting batch size and using mixed-precision (`fp16`) training. |
| **7. Chatting with the Fine-Tuned Model** | Interacted with the fine-tuned model using a text-generation pipeline. Sample prompts like “Explain Paracetamol” or “Tell me about Botulism” were given to observe the model’s medical knowledge. |

---

##  Model Details

- **Base model:** `aboonaji/llama2finetune-v2`  
- **Architecture:** LLaMA 2 (Causal LM)  
- **Framework:** Hugging Face Transformers + TRL (SFT Trainer)  
- **Quantization:** 4-bit (BitsAndBytes)  
- **Dataset:** `aboonaji/wiki_medical_terms_llam2_format` (medical text)  
- **Environment:** Google Colab GPU  
- **Libraries used:** `torch`, `transformers`, `trl`, `datasets`, `peft`, `bitsandbytes`

---

##  Methodology

1. **Model Preparation** – Load pre-trained model and tokenizer from Hugging Face.  
2. **Fine-Tuning Setup** – Configure LoRA (PEFT) and training parameters.  
3. **Dataset Loading** – Fetch medical dataset directly from Hugging Face.  
4. **Training** – Fine-tune model on the dataset using SFT Trainer.  
5. **Evaluation / Chat** – Interact with the fine-tuned model using sample prompts.

---
## Optional

If you ever want to reproduce the same environment locally or in another Colab:
```
pip install -r requirements.txt
```
---

## Example Interaction

User: Explain Paracetamol.
Model: Paracetamol (acetaminophen) is a common pain reliever and fever reducer. It works by inhibiting prostaglandin synthesis in the brain, which helps reduce pain and temperature.

---

##  Tools Used

- **Google Colab** — for training and execution  
- **Hugging Face Hub** — for model and dataset hosting  
- **PyTorch** — backend framework for model computation  
- **Transformers & TRL** — training and text generation utilities  
- **BitsAndBytes** — for quantized model loading (4-bit)  
- **PEFT (LoRA)** — for parameter-efficient fine-tuning

---

##  Output

- Fine-tuned LLaMA 2 model with enhanced understanding of **medical terminology and context**.  
- Model responds to health-related prompts more accurately than the base version.  
- Results verified through manual chat testing in Colab.

---

##  Notes

- Training was done entirely in **Google Colab**, not in VS Code or a local setup.  
- **No API keys or tokens** are included in this repository.  
- Model and dataset are **publicly available** on Hugging Face Hub.  
- This repository only contains the **notebook (.ipynb)** and documentation for reproducibility.

---

## My Website

- https://bharathyalagi.netlify.app/

# Thank You 
