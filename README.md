# RAGEmini
play on RAGE + gemini as MVP
MVP to connect with ollama recognizing ollama from localhost if running<br />
MMP to connect with gemini API including experimental from toggle<br /><br />
<b>RAGE</b> folder contains the as yet to be implented <b>R</b>etrieval <b>A</b>ugmented <b>G</b>enerative <b>E</b>ngine<br />
functions as a basic UI for chat response from localhost ollama model or Gemini<br />
with memory.py and logger.py<br />

```txt
RAGEmini/
├── src/
│   ├── memory.py
│   ├── logger.py
│   ├── openmind.py
│   └── locallama.py
├── gfx/
│   └── styles.css
├── memory/
│   ├── sessions/
│   ├── knowledge/
│   └── long_term_memory.json
└── rage.py
```
# RAGE + GEMINI == RAGEmini
Retrieval Augmented Generative Engine<br />
<br />

RAGE Retrieval Augmented Generative Engine is a dynamic engine designed to learn from context, injest and memory over time.<br />

# Context-Aware Responses: 
By leveraging the continuously updated data and learning from past interactions, RAGE can understand and respond to nuances in user queries. This ability makes it particularly effective in scenarios where context heavily influences the nature of the response.
# Adaptive Response Generation: 
As RAGE evolves, it becomes more adept at predicting user needs and adjusting its responses accordingly, ensuring high relevance and personalization<br />
perform manual INSTALL or INSTALL not both<br />
```python
git clone https://github.com/GATERAGE/DeepSeekRAGE
```
```python
python3.11 -m venv rage
```
```python
source rage/bin/activate
```
```python
pip install --no-cache-dir -r requirements.txt
```

```python
streamlit run rage.py
```

INSTALL<br />
```bash
source install.sh
```

```python
streamlit run rage.py
```
