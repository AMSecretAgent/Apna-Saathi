# Apna Saathi
### Making Digital India Accessible to Every Household

**An Offline, Voice-First AI Assistant for Digital Inclusion in Rural and Underserved India**

**Hack for Social Cause (HSC) 2027 | Digital Inclusion**

Organised through the **MY Bharat Portal** by the **Department of Youth Affairs, Ministry of Youth Affairs and Sports**, as part of the **Viksit Bharat Young Leaders' Dialogue 2027**, with **IIT Bombay as the National Knowledge Partner**.

---

## The Problem

India's digital transformation has brought government services, healthcare information, and essential resources online. However, access to these services remains unequal.

For millions of people, particularly in rural and underserved communities, digital inclusion is not simply about having an internet connection.

It is about being able to use technology independently.

A farmer may not know how to check eligibility for PM-KISAN. An elderly person may struggle to navigate an online pension portal. A homemaker may need basic health information but find available resources difficult to understand.

The barriers are often fundamental:

- **Digital literacy:** Many citizens are unfamiliar with applications, websites, and online procedures.
- **Language:** Essential information is frequently difficult to access in familiar regional languages.
- **Connectivity:** Internet access can be unreliable, expensive, or unavailable.
- **Accessibility:** Text-heavy interfaces exclude people with limited literacy or difficulty using conventional digital devices.
- **Dependence on intermediaries:** People may need others to access information that should be available to them directly.

**The challenge is not merely to bring people online. It is to make digital services usable even when they are offline.**

## Our Solution

**Apna Saathi** is a compact, offline, voice-first AI household assistant designed to make essential information accessible through natural conversation.

Instead of navigating applications, typing queries, or understanding complicated digital interfaces, users simply ask a question aloud in Hindi and receive a spoken response in Hindi.

No internet connection. No smartphone application. No account creation. No personal information sent to a remote server.

Built on the open-source **Suno Sutra platform**, Apna Saathi combines speech recognition, translation, local information retrieval, and an on-device language model to provide short, understandable, grounded answers.

### How it works for a user

**Ask:** "PM-KISAN ka paisa kitna milta hai?"

**Understand:** The device recognises the Hindi question, identifies the relevant government scheme, and retrieves information from its local knowledge base.

**Respond:** The assistant provides a concise spoken answer in Hindi, based on the retrieved information.

The entire process runs locally on the device.

---

## Digital Inclusion Through Apna Saathi

| Barrier | Our Approach |
|---|---|
| Limited reading and writing ability | Voice-first interaction |
| Language barriers | Hindi speech input and output with a translation-based architecture |
| Unreliable internet | Fully offline inference |
| Lack of smartphones | Standalone hardware device |
| Complex government information | Simplified, grounded explanations |
| Privacy concerns | On-device processing without transmitting queries |
| Limited computing resources | Lightweight language model and SQLite retrieval |

Our goal is to reduce the technological knowledge required to access essential information.

## Core Use Cases

### 1. Government Welfare and Citizen Services

Apna Saathi helps citizens understand public welfare schemes and basic eligibility, benefits, and application-related information.

Current knowledge domains include:

- PM-KISAN
- Ayushman Bharat
- Ration cards and food security
- Old-age and other pensions
- Pradhan Mantri Ujjwala Yojana

**Example questions:**

- "PM-KISAN ke liye kaun eligible hai?"
- "Ayushman card kaise banega?"
- "Ration card banwane ke liye kya chahiye?"
- "Buzurgon ko pension kaise milti hai?"

The assistant retrieves relevant information from its local knowledge base rather than relying solely on the language model's memory.

### 2. Family Health Awareness

Apna Saathi provides basic health awareness and helps families understand when professional medical assistance may be necessary.

Supported information areas include:

- Oral rehydration and dehydration awareness
- General medicine-timing guidance
- Recognition of warning signs
- Public healthcare facilities and free checkups
- When to consult a medical professional

**Example questions:**

- "ORS kab dena chahiye?"
- "Bukhar mein doctor ko kab dikhana chahiye?"
- "Sarkari hospital mein free checkup kahan hota hai?"

**Safety first:** Apna Saathi is not a diagnostic or prescription system. Requests involving medical diagnosis or specific medicine dosages are redirected to qualified healthcare professionals or appropriate helplines.

---

## What Makes Apna Saathi Different?

### Offline by Design

Unlike cloud-dependent AI assistants, Apna Saathi is designed to operate without continuous internet connectivity.

Speech processing, translation, retrieval, language generation, and speech synthesis run through local services and models.

### Voice Instead of Screens

The interaction model is intentionally simple:

**Press and hold → Speak → Release → Listen**

This reduces dependence on typing, reading, and navigating unfamiliar interfaces.

### Grounded Responses

Small language models can generate plausible but incorrect answers.

Apna Saathi addresses this by retrieving relevant information from a curated local knowledge base before generating a response.

When sufficiently relevant information is unavailable, the assistant is designed to acknowledge that limitation rather than invent an answer.

### Resource-Efficient Architecture

The system uses:

- A 1.5B-parameter language model
- SQLite FTS5 for lightweight information retrieval
- Local BHASHINI-compatible speech and translation services
- Short, bounded answer generation

This architecture is designed to operate within the memory and compute constraints of compact edge hardware.

### Privacy by Architecture

Questions are processed locally.

The runtime does not require user registration, cloud inference, or transmission of household queries to external servers.

---

## Technical Architecture

```text
           USER SPEAKS IN HINDI
                    |
                    v
          Speech Recognition
           BHASHINI / ASR
                    |
                    v
           Hindi Transcript
                    |
                    v
          Machine Translation
           Hindi -> English
                    |
                    v
           SQLite FTS5
       Local Knowledge Retrieval
                    |
                    v
          Grounded LLM Inference
            Qwen2.5-1.5B
                    |
                    v
           Answer Formatting
             textshape.py
                    |
                    v
          Machine Translation
           English -> Hindi
                    |
                    v
           Text-to-Speech
           BHASHINI / TTS
                    |
                    v
         SPOKEN ANSWER IN HINDI
```

### Why Translate Before Reasoning?

Apna Saathi uses an English-centred reasoning pipeline.

Hindi speech is transcribed and translated into English. The language model receives the translated question and relevant English grounding passages, generates a concise answer, and the response is translated back into Hindi.

This approach allows the system to use a compact model for short English instruction-following while relying on specialised language components for speech and translation.

It also provides an architectural path towards additional Indian languages without requiring a substantially larger language model.

## Technology Stack

| Component | Technology |
|---|---|
| Edge computing | NVIDIA Jetson Orin Nano |
| Application | Python |
| Speech recognition | BHASHINI-compatible local ASR |
| Translation | BHASHINI-compatible local NMT |
| Language model | Qwen2.5-1.5B through Ollama |
| Retrieval | SQLite FTS5 |
| Speech synthesis | BHASHINI-compatible local TTS |
| Knowledge base | Curated TSV corpus |
| Answer shaping | Custom Python text-processing pipeline |
| Deployment | Shell scripts and systemd service |

---

## Getting Started

### Run Without Hardware

The retrieval and evaluation components can run on a laptop with Python 3.

```bash
git clone https://github.com/AMSecretAgent/Apna-Saathi.git
cd Apna-Saathi
./setup.sh
```

The setup script fetches the Suno Sutra platform source, applies the necessary patches, overlays the Apna Saathi application, and runs self-checks.

### Run Local Self-Checks

```bash
python3 pocketinfer/retrieval.py
python3 pocketinfer/textshape.py
python3 tools/eval_apna_saathi.py --retrieval-only -v
python3 tools/eval_apna_saathi.py --sweep-threshold
```

These checks do not require a Jetson board, downloaded language models, or an internet connection.

### Deploy to Jetson

With a device accessible through SSH or Tailscale:

```bash
export DEV=ubuntu@<name-or-ip>
less docs/BOARD_BRINGUP.md
./tools/deploy.sh "$DEV"
```

Follow the ten-step bring-up guide in `docs/BOARD_BRINGUP.md` before deployment.

---

## Repository Structure

```text
Apna-Saathi/
|
|-- setup.sh
|
|-- pocketinfer/
|   |-- applications/
|   |   `-- apna_saathi.py
|   |-- retrieval.py
|   |-- textshape.py
|   `-- data/
|       |-- corpus.tsv
|       `-- eval.jsonl
|
|-- tools/
|   |-- eval_apna_saathi.py
|   |-- probe_bhashini.py
|   |-- make_canned.py
|   |-- deploy.sh
|   `-- bhashini_cloud.py
|
|-- patches/
|
|-- docs/
|   |-- BOARD_BRINGUP.md
|   |-- ARCHITECTURE.md
|   |-- BOM.md
|   |-- BUSINESS_PLAN.md
|   `-- VIDEO_SCRIPT.md
|
|-- THIRD_PARTY_NOTICES.md
|
`-- vendor/
    `-- suno-sutra-sw
```

## Knowledge Base and Retrieval

Apna Saathi uses a locally stored, curated knowledge base.

Each entry in `pocketinfer/data/corpus.tsv` contains:

| Field | Purpose |
|---|---|
| `doc_id` | Stable document identifier |
| `section` | Scheme, health, or civic category |
| `title` | Human-readable source title |
| `aliases` | Hindi, English, and romanised query variants |
| `body` | Concise grounding paragraph |

The retrieval system uses SQLite FTS5 rather than a separate embedding model, reducing memory requirements.

To evaluate corpus coverage:

```bash
python3 tools/eval_apna_saathi.py --retrieval-only -v
```

Retrieval evaluation measures coverage against reference questions. It does not represent end-to-end accuracy for real users, where speech recognition, translation, background noise, and other factors can affect performance.

## Evaluation and Reliability

The project includes tools for evaluating retrieval performance, response latency, and safety behaviour.

```bash
# Repeated evaluation with Markdown output
python3 tools/eval_apna_saathi.py --repeat 3 --markdown

# Microphone gain and ASR evaluation
python3 tools/eval_apna_saathi.py --sweep-gain

# Retrieval threshold calibration
python3 tools/eval_apna_saathi.py --sweep-threshold
```

The evaluation harness distinguishes retrieval-only testing from full audio-path performance.

Medical safety checks are also included to test whether unsupported dosage and diagnosis questions are appropriately refused or redirected.

See [Technical Architecture](https://github.com/AMSecretAgent/Apna-Saathi/blob/main/docs/ARCHITECTURE.md) for details.

---

## Social Impact and Future Scope

Apna Saathi is designed around a simple principle: **essential information should not require advanced digital skills.**

The project targets accessibility for people who may be excluded by conventional smartphone- and internet-dependent services, including rural households, older adults, first-time technology users, and people with limited literacy.

Potential future extensions include:

- Additional Indian and regional languages
- Expanded government scheme coverage
- State- and district-specific public service information
- Improved accessibility for elderly users
- Community deployments through local assistance centres
- Field testing with households unfamiliar with digital interfaces
- Improved update and verification processes for offline knowledge

These are proposed directions rather than claims about the current implementation.

## Alignment with HSC 2027: Digital Inclusion

Apna Saathi directly addresses digital exclusion by removing several barriers between citizens and essential information.

**Accessibility:** People can interact through speech instead of navigating text-heavy applications.

**Affordability:** Shared, standalone edge devices can provide an alternative to requiring every household member to own a smartphone.

**Connectivity independence:** Core functions are designed to work without internet access.

**Language inclusion:** Hindi interaction is supported, with an architecture intended to accommodate more Indian languages.

**Digital empowerment:** Citizens can independently seek information about welfare schemes and basic health awareness.

**Privacy:** Household queries remain on the device.

The broader vision is to make AI a practical public-interest technology, accessible beyond digitally connected and technologically confident populations.

---




**Apna Saathi — Because access to information should not depend on literacy, language, connectivity, or the ability to operate a smartphone.**
