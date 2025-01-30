# **BERT-based Text Classification**

**Developed by:** Karan Bhosle  
**Contact:** [LinkedIn - Karan Bhosle](https://www.linkedin.com/in/karanbhosle/)  

### **Project Description:**
This project demonstrates the use of the **BERT (Bidirectional Encoder Representations from Transformers)** model for text classification. BERT, a transformer-based model, is pre-trained on a large corpus of text data and fine-tuned on a smaller, task-specific dataset for the task of text classification. In this case, the model classifies text into two categories, indicated by the binary labels `0` and `1`.

The project involves the following:
1. Preprocessing text data using the **BERT Tokenizer**.
2. Fine-tuning a pre-trained **BERT model** on a small custom dataset.
3. Using **PyTorch** for model training and evaluation.
4. Evaluating the model using **accuracy** metrics.

### **Objective:**
- Fine-tune the **BERT** model to classify texts into two categories (binary classification).
- Implement a custom dataset class for tokenizing and batching the data.
- Train the model for a few epochs, and evaluate the performance on a test set.
- Output the accuracy score of predictions.

---

### **Tools and Libraries Used:**
- **Python**: Programming language
- **PyTorch**: Deep learning framework used for model training.
- **Transformers**: Library by Hugging Face, used to load and fine-tune pre-trained transformer models like BERT.
- **Scikit-learn**: Provides utilities for performance metrics like accuracy.

---

### **Steps Involved:**

#### **1. Install Libraries:**
First, necessary libraries are installed:
```bash
!pip install transformers torch
```

#### **2. Import Libraries:**
Importing required libraries such as **torch**, **transformers**, and **sklearn**.
```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
```

#### **3. Prepare Dataset:**
A small dataset of texts and labels is created. These are example texts that will be used for training.
```python
texts = [
    "This is an example sentence.",
    "Each sentence is a separate text.",
    "We will process them using BERT.",
    "BERT is a powerful language model.",
    "It can perform various NLP tasks.",
    "Such as sentiment analysis, text classification.",
    "And question answering.",
    "We will use a pre-trained BERT model.",
    "And fine-tune it on this small dataset.",
    "This is the last example sentence."]
    
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Binary labels
```

#### **4. Dataset Class:**
A custom dataset class `SequentialDataset` is implemented that tokenizes the input texts using the **BERT Tokenizer** and prepares them for model training.
```python
class SequentialDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoded_inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids = encoded_inputs['input_ids'].squeeze()
        attention_mask = encoded_inputs['attention_mask'].squeeze()
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label
```

#### **5. Tokenization and Batching:**
The dataset is tokenized using BERT’s tokenizer and converted into batches using **DataLoader** for efficient training.
```python
dataset = SequentialDataset(texts, labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
```

#### **6. Model Setup:**
A pre-trained BERT model (`bert-base-uncased`) is loaded and fine-tuned for sequence classification with two output labels (binary classification).
```python
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
```

#### **7. Training Setup:**
The **AdamW** optimizer is used to optimize the model during training. The model is then trained for the specified number of epochs.
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
```

#### **8. Training Loop:**
The model is trained for **EPOCHS** iterations. For each batch, the loss is computed and backpropagated to update the model weights.
```python
for epoch in range(EPOCHS):
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        # Move tensors to GPU if available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass and compute loss
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item()}")
```

#### **9. Evaluation:**
After training, the model is evaluated on a test set. Predictions are made using the model, and accuracy is computed using **scikit-learn**'s **accuracy_score** function.
```python
accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

#### **10. Test Set:**
A small test set of texts is prepared for evaluation. The model's predictions are compared to the actual labels, and accuracy is printed.
```python
test_texts = [
    "BERT is a powerful language model.",
    "It can perform various NLP tasks.",
    "Such as sentiment analysis, text classification.",
    "And question answering.",
    "We will use a pre-trained BERT model."]

test_labels = [0, 1, 0, 1, 0]
```

---

### **Model Evaluation:**
The final evaluation step computes the model's accuracy on the test set. The predicted labels are compared to the true labels and the accuracy score is printed.
```python
print(f'Predictions: {predictions}')
print(f'Labels: {test_labels}')
```

---

### **Conclusion:**
This project demonstrates how to fine-tune a pre-trained **BERT model** for binary text classification tasks. The model is trained and evaluated on a small dataset, and the final accuracy is computed using **scikit-learn's** **accuracy_score** function.

---

### **References:**
1. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**: https://arxiv.org/abs/1810.04805
2. **Hugging Face Transformers Documentation**: https://huggingface.co/docs/transformers
3. **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
