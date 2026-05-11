# Guía de Fine-Tuning para el Extractor NER Quipux

## Entidades 1 (BETO / Persona) y 10 (textcat / Acción Requerida)

\---

## 1\. Fine-Tuning de BETO para Entidad "Persona" (Entidad 1)

### Por qué es necesario

BETO (`dccuchile/bert-base-spanish-wwm-cased`) está pre-entrenado en español
general. Los oficios Quipux contienen patrones específicos que el modelo base
no captura correctamente:

* Tratamientos profesionales pegados al nombre: *Abg., Ing., Lcda., Dr., MSc.*
* Nombres compuestos de 3 o 4 palabras: *María de los Ángeles Vega Morales*
* Contexto burocrático: *"suscribe la presente, Ing. Juan Pérez, Director…"*

### Paso 1 — Preparar el corpus de anotación

Formato requerido: **CoNLL-2003** (una token por línea, etiqueta BIO al final).

```
Ing.        B-PER
María       I-PER
de          I-PER
los         I-PER
Ángeles     I-PER
Vega        I-PER
,           O
Director    O
```

**Volumen mínimo recomendado:** 500 oficios anotados (\~8.000–12.000 tokens PER).
**Herramienta de anotación:** Label Studio (gratuito, self-hosted).

```bash
pip install label-studio
label-studio start
```

Crear proyecto con template "Named Entity Recognition", exportar en formato CoNLL.

### Paso 2 — Script de fine-tuning

```python
# fine\\\\\\\_tune\\\\\\\_beto\\\\\\\_ner.py
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification,
)
from datasets import load\\\\\\\_dataset
import numpy as np
from seqeval.metrics import f1\\\\\\\_score, classification\\\\\\\_report

# Etiquetas BIO para el dominio Quipux
LABEL\\\\\\\_LIST = \\\\\\\["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-CARGO", "I-CARGO"]
LABEL2ID = {l: i for i, l in enumerate(LABEL\\\\\\\_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

MODEL\\\\\\\_CHECKPOINT = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from\\\\\\\_pretrained(MODEL\\\\\\\_CHECKPOINT)

# Cargar dataset en formato CoNLL desde carpeta /data/conll/
dataset = load\\\\\\\_dataset("conll2003", data\\\\\\\_files={
    "train": "data/conll/train.conll",
    "validation": "data/conll/dev.conll",
})

def tokenize\\\\\\\_and\\\\\\\_align\\\\\\\_labels(examples):
    tokenized = tokenizer(
        examples\\\\\\\["tokens"],
        truncation=True,
        is\\\\\\\_split\\\\\\\_into\\\\\\\_words=True,
        max\\\\\\\_length=512,
    )
    labels = \\\\\\\[]
    for i, label in enumerate(examples\\\\\\\["ner\\\\\\\_tags"]):
        word\\\\\\\_ids = tokenized.word\\\\\\\_ids(batch\\\\\\\_index=i)
        prev\\\\\\\_word\\\\\\\_idx = None
        label\\\\\\\_ids = \\\\\\\[]
        for word\\\\\\\_idx in word\\\\\\\_ids:
            if word\\\\\\\_idx is None:
                label\\\\\\\_ids.append(-100)
            elif word\\\\\\\_idx != prev\\\\\\\_word\\\\\\\_idx:
                label\\\\\\\_ids.append(label\\\\\\\[word\\\\\\\_idx])
            else:
                label\\\\\\\_ids.append(-100)  # ignorar sub-tokens
            prev\\\\\\\_word\\\\\\\_idx = word\\\\\\\_idx
        labels.append(label\\\\\\\_ids)
    tokenized\\\\\\\["labels"] = labels
    return tokenized

tokenized\\\\\\\_dataset = dataset.map(tokenize\\\\\\\_and\\\\\\\_align\\\\\\\_labels, batched=True)

model = AutoModelForTokenClassification.from\\\\\\\_pretrained(
    MODEL\\\\\\\_CHECKPOINT,
    num\\\\\\\_labels=len(LABEL\\\\\\\_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)

args = TrainingArguments(
    output\\\\\\\_dir="./models/beto-quipux-ner",
    num\\\\\\\_train\\\\\\\_epochs=5,           # 3–5 épocas suficiente para corpus pequeño
    per\\\\\\\_device\\\\\\\_train\\\\\\\_batch\\\\\\\_size=16,
    per\\\\\\\_device\\\\\\\_eval\\\\\\\_batch\\\\\\\_size=16,
    learning\\\\\\\_rate=2e-5,           # LR estándar para fine-tuning BERT
    weight\\\\\\\_decay=0.01,
    evaluation\\\\\\\_strategy="epoch",
    save\\\\\\\_strategy="epoch",
    load\\\\\\\_best\\\\\\\_model\\\\\\\_at\\\\\\\_end=True,
    metric\\\\\\\_for\\\\\\\_best\\\\\\\_model="f1",
    push\\\\\\\_to\\\\\\\_hub=False,            # True si quieres subir a HuggingFace Hub
)

def compute\\\\\\\_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true\\\\\\\_labels = \\\\\\\[\\\\\\\[ID2LABEL\\\\\\\[l] for l in label if l != -100] for label in labels]
    true\\\\\\\_preds = \\\\\\\[
        \\\\\\\[ID2LABEL\\\\\\\[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    return {"f1": f1\\\\\\\_score(true\\\\\\\_labels, true\\\\\\\_preds)}

trainer = Trainer(
    model=model,
    args=args,
    train\\\\\\\_dataset=tokenized\\\\\\\_dataset\\\\\\\["train"],
    eval\\\\\\\_dataset=tokenized\\\\\\\_dataset\\\\\\\["validation"],
    tokenizer=tokenizer,
    data\\\\\\\_collator=DataCollatorForTokenClassification(tokenizer),
    compute\\\\\\\_metrics=compute\\\\\\\_metrics,
)

trainer.train()
trainer.save\\\\\\\_model("./models/beto-quipux-ner/final")
print(classification\\\\\\\_report(true\\\\\\\_labels, true\\\\\\\_preds))
```

### Paso 3 — Integrar en quipux\_extractor.py

```python
# En EntityExtractor.\\\\\\\_\\\\\\\_init\\\\\\\_\\\\\\\_(), reemplazar la línea de beto\\\\\\\_ner:
self.beto\\\\\\\_ner = pipeline(
    "ner",
    model="./models/beto-quipux-ner/final",   # ← checkpoint finze-tuned
    tokenizer="./models/beto-quipux-ner/final",
    aggregation\\\\\\\_strategy="simple",
    device=-1,
)
```

\---

## 2\. Fine-Tuning de spaCy textcat para Entidad "Acción Requerida" (Entidad 10)

### Por qué es necesario

El clasificador `textcat` no existe por defecto: es un componente que se
entrena desde cero sobre el corpus Quipux con 4 categorías:

|Clase|Descripción|Ejemplo de trigger|
|-|-|-|
|`SOLICITUD`|El emisor pide algo|*"Solicito a usted se sirva…"*|
|`DISPOSICIÓN`|El emisor ordena/dispone|*"Dispongo se proceda a…"*|
|`INFORME`|El documento informa/reporta|*"Informo a usted que…"*|
|`AUTORIZACIÓN`|Se concede o pide permiso|*"Autorizo la ejecución de…"*|

**Volumen mínimo recomendado:** 200 oficios con clase etiquetada manualmente.

### Paso 1 — Preparar datos de entrenamiento

```python
# Formato esperado por spaCy textcat
TRAIN\\\\\\\_DATA = \\\\\\\[
    ("Solicito a usted se sirva proporcionar la información requerida.",
     {"cats": {"SOLICITUD": 1.0, "DISPOSICIÓN": 0.0, "INFORME": 0.0, "AUTORIZACIÓN": 0.0}}),
    ("Dispongo que el Director Distrital proceda con el trámite.",
     {"cats": {"SOLICITUD": 0.0, "DISPOSICIÓN": 1.0, "INFORME": 0.0, "AUTORIZACIÓN": 0.0}}),
    ("Informo a usted que el proceso ha sido atendido.",
     {"cats": {"SOLICITUD": 0.0, "DISPOSICIÓN": 0.0, "INFORME": 1.0, "AUTORIZACIÓN": 0.0}}),
    # ... 200+ ejemplos
]
```

### Paso 2 — Script de entrenamiento

```python
# fine\\\\\\\_tune\\\\\\\_textcat.py
import spacy
from spacy.training import Example
import random
from pathlib import Path

OUTPUT\\\\\\\_DIR = Path("./models/textcat-quipux")

nlp = spacy.blank("es")

# Añadir componente textcat
textcat = nlp.add\\\\\\\_pipe("textcat")
for label in \\\\\\\["SOLICITUD", "DISPOSICIÓN", "INFORME", "AUTORIZACIÓN"]:
    textcat.add\\\\\\\_label(label)

# Convertir datos al formato Example
examples = \\\\\\\[]
for text, annotations in TRAIN\\\\\\\_DATA:
    doc = nlp.make\\\\\\\_doc(text)
    examples.append(Example.from\\\\\\\_dict(doc, annotations))

# Entrenamiento
optimizer = nlp.initialize()
EPOCHS = 20
for epoch in range(EPOCHS):
    random.shuffle(examples)
    losses = {}
    for batch in spacy.util.minibatch(examples, size=8):
        nlp.update(batch, sgd=optimizer, losses=losses)
    if epoch % 5 == 0:
        print(f"Epoch {epoch} — Loss: {losses\\\\\\\['textcat']:.4f}")

OUTPUT\\\\\\\_DIR.mkdir(parents=True, exist\\\\\\\_ok=True)
nlp.to\\\\\\\_disk(OUTPUT\\\\\\\_DIR)
print(f"Modelo guardado en {OUTPUT\\\\\\\_DIR}")
```

### Paso 3 — Integrar en quipux\_extractor.py

Una vez entrenado, descomentar el bloque en `extract\\\\\\\_accion()`:

```python
# En EntityExtractor.\\\\\\\_\\\\\\\_init\\\\\\\_\\\\\\\_():
textcat\\\\\\\_nlp = spacy.load("./models/textcat-quipux")
self.nlp.add\\\\\\\_pipe(
    "textcat",
    source=textcat\\\\\\\_nlp,
    last=True,
)

# En extract\\\\\\\_accion(), descomentar:
if "textcat" in self.nlp.pipe\\\\\\\_names and accion\\\\\\\_texto:
    sub\\\\\\\_doc = self.nlp(accion\\\\\\\_texto)
    scores = sub\\\\\\\_doc.cats
    if scores:
        accion\\\\\\\_clase = max(scores, key=scores.get)
        if scores\\\\\\\[accion\\\\\\\_clase] < 0.6:
            accion\\\\\\\_clase = None
```

\---

## 3\. Checklist de producción

```
\\\\\\\[ ] Anotar corpus BETO (500 oficios, formato CoNLL, Label Studio)
\\\\\\\[ ] Anotar corpus textcat (200 oficios, 4 clases)
\\\\\\\[ ] Ejecutar fine\\\\\\\_tune\\\\\\\_beto\\\\\\\_ner.py → F1 objetivo ≥ 0.88
\\\\\\\[ ] Ejecutar fine\\\\\\\_tune\\\\\\\_textcat.py → F1 objetivo ≥ 0.85
\\\\\\\[ ] Ampliar CATALOG\\\\\\\_ORGS con lista completa SNAP (descarga oficial)
\\\\\\\[ ] Ampliar CATALOG\\\\\\\_CARGOS con catálogo MRL (descarga oficial)
\\\\\\\[ ] Ampliar GAZETTEER\\\\\\\_EC con todas las parroquias del IGM
\\\\\\\[ ] Validar pipeline completo sobre 50 oficios reales no vistos
\\\\\\\[ ] Versionar modelos fine-tuned en repositorio de tesis (Git LFS)
```

