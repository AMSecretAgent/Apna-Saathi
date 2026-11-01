# Apna Saathi
### Offline, Multilingual AI for Accessible Communities

**Making essential information accessible, regardless of language, literacy, or connectivity.**

## Overview

Apna Saathi is an offline-first, voice-enabled AI assistant designed to bridge the digital divide in rural and underserved communities.

Millions of people face challenges accessing essential information due to language barriers, limited digital literacy, unreliable internet connectivity, and complex digital interfaces.

Apna Saathi addresses these challenges through a multilingual, conversational interface that allows users to interact with technology naturally, without depending on smartphones, continuous internet access, or advanced technical knowledge.

## The Problem

Despite the expansion of digital services, accessibility remains a significant challenge for underserved communities.

Common barriers include:

- Limited or unreliable internet connectivity
- Low digital and technological literacy
- Language barriers in accessing digital information
- Complex government service procedures
- Difficulty finding understandable health and welfare information
- Dependence on intermediaries for basic digital assistance

## Our Solution

Apna Saathi brings AI-powered assistance directly to communities through an offline-capable, voice-first system.

Users can ask questions verbally and receive responses in accessible language. The system combines speech recognition, multilingual processing, information retrieval, local AI inference, and speech synthesis.

Its architecture is designed for deployment on compact edge-computing devices.

## Key Features

| Feature | Description |
|---|---|
| Offline-first operation | Supports local processing without continuous internet connectivity |
| Voice-first interaction | Reduces dependence on reading, typing, and complex interfaces |
| Multilingual assistance | Supports Hindi-oriented workflows and additional Indian-language capabilities |
| Government welfare guidance | Provides information about public schemes and benefits |
| Health information | Offers basic health-related guidance with safety considerations |
| OCR capabilities | Extracts text from camera input |
| Edge AI | Runs supported inference workloads locally |
| Information retrieval | Retrieves relevant information from structured local knowledge resources |
| Safety mechanisms | Includes safeguards for sensitive queries and response handling |

## System Architecture

```text
             USER
               |
       Voice / Camera Input
               |
       +-------+-------+
       |               |
 Speech Recognition   OCR
       |               |
       +-------+-------+
               |
       Language Processing
               |
       Information Retrieval
               |
         Local AI Model
               |
       Safety and Validation
               |
       Speech / Display Output
               |
             USER
```

### Core Modules

| Component | File |
|---|---|
| Main application | `python/pocketinfer/applications/apna_saathi.py` |
| Audio processing | `python/pocketinfer/audio.py` |
| Speech and language models | `python/pocketinfer/models/` |
| OCR | `python/pocketinfer/ocr.py` |
| Vision processing | `python/pocketinfer/vision.py` |
| Knowledge retrieval | `python/pocketinfer/retrieval.py` |
| Safety handling | `python/pocketinfer/safety.py` |
| Hardware interface | `python/pocketinfer/boards/` |
| User interface | `python/pocketinfer/ui/` |

## Repository Structure

```text
Apna-Saathi/
├── python/
│   ├── pocketinfer/
│   │   ├── applications/
│   │   ├── boards/
│   │   ├── models/
│   │   ├── data/
│   │   ├── ui/
│   │   ├── audio.py
│   │   ├── ocr.py
│   │   ├── retrieval.py
│   │   ├── safety.py
│   │   └── vision.py
│   ├── tools/
│   ├── requirements.txt
│   └── setup.py
├── ioexpander/
├── rootfs/
├── assets/
├── LICENSE
└── README.md
```

## Installation and Setup

### Requirements

- Python 3 with a compatible environment
- Required Python dependencies
- Compatible microphone and speaker
- Supported edge-computing hardware for device deployment
- Appropriate speech, language, and inference models
- Hardware-specific drivers and services where applicable

### Clone the Repository

```bash
git clone https://github.com/AMSecretAgent/Apna-Saathi.git
cd Apna-Saathi
```

### Set Up the Python Environment

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate using:

```cmd
.venv\Scripts\activate
```

### Hardware Deployment

The project includes device integration and provisioning resources under `python/pocketinfer/boards/` and `rootfs/`.

Hardware deployment requires the appropriate device configuration, model assets, runtime dependencies, and peripheral setup.

Large pretrained model archives are not bundled with the source repository. A fully operational deployment may require additional model installation and configuration.

## Sample Data

The repository includes:

- `python/pocketinfer/data/corpus.tsv` — structured knowledge corpus for retrieval
- `python/pocketinfer/data/eval.jsonl` — evaluation examples

These files support experimentation, retrieval testing, and application evaluation.

## Testing and Evaluation

The repository contains development and evaluation utilities:

| File | Purpose |
|---|---|
| `python/tools/eval_apna_saathi.py` | Application evaluation |
| `python/tools/make_canned.py` | Preparation of test resources |
| `python/tools/probe_bhashini.py` | Language-service diagnostics |
| `python/tools/watch_queries.py` | Query monitoring and debugging |

The exact execution options and expected results depend on the installed runtime, available models, and script configuration.

Testing areas include retrieval relevance, multilingual processing, response safety, and device interaction.

## Accessibility and Social Impact

Apna Saathi is designed around the principle that access to useful technology should not depend on someone's ability to read, speak English, own a smartphone, or maintain a reliable internet connection.

Its potential applications include:

- Rural household assistance
- Community information centres
- Assisted access to government services
- Multilingual public-information delivery
- Digital literacy support
- Offline information access in connectivity-constrained environments

## Limitations and Responsible Use

Apna Saathi provides informational assistance and does not replace qualified medical professionals, official government authorities, or emergency services.

Responses concerning healthcare, eligibility, or government procedures should be verified through appropriate authoritative sources.

Actual offline performance, supported languages, and response quality depend on the deployed hardware and models.

## Open Source and License

This repository contains open-source application code and adapted third-party components.

See `LICENSE` for the applicable license. Original copyright notices and legally required third-party attributions remain applicable.

## Vision

**Technology should adapt to people, not the other way around.**

Apna Saathi aims to make artificial intelligence more accessible through multilingual, offline-first, human-centred interaction.

---

**Project:** Apna Saathi  
**Focus:** Offline AI · Multilingual Accessibility · Digital Inclusion · Edge Computing  
**Repository:** https://github.com/AMSecretAgent/Apna-Saathi
