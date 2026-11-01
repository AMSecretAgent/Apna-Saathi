# Apna Saathi
### Offline, Multilingual AI for Inclusive Rural Communities

**Think Global, Hack Local — Technology for Social Transformation**

Apna Saathi is an offline-first, voice-enabled AI assistant designed to make essential information accessible to underserved and rural communities in India.

It addresses barriers such as limited internet connectivity, language differences, low digital literacy, and the complexity of navigating public services.

The goal is simple: **Make technology accessible to everyone, regardless of language, literacy, location, or connectivity.**

## The Problem

Millions of people face challenges accessing essential digital services due to:

- Unreliable or unavailable internet connectivity
- Limited familiarity with smartphones and digital interfaces
- Language and literacy barriers
- Difficulty understanding government welfare schemes
- Limited access to reliable, understandable health information

Apna Saathi explores how locally deployed AI can help bridge these gaps through accessible, multilingual, voice-based interaction.

## Our Solution

Apna Saathi provides a conversational interface through which users can ask questions in their preferred language and receive spoken responses without requiring continuous internet access.

The system is designed for deployment on compact edge-computing hardware, making it suitable for community centres, rural households, and assisted-service environments.

## Key Features

| Feature | Description |
|---|---|
| Offline-first AI | Processes supported requests locally without continuous cloud connectivity |
| Voice interaction | Enables interaction without typing or navigating complex menus |
| Multilingual support | Designed for Hindi and additional Indian-language workflows |
| Welfare assistance | Provides guidance on government schemes and public benefits |
| Health information | Offers basic informational guidance and encourages appropriate medical care |
| OCR and visual assistance | Supports camera-based text recognition |
| Edge deployment | Designed for compact computing devices |
| Safety mechanisms | Includes application-level safeguards and response handling |

## System Architecture

The application combines audio processing, language models, retrieval, vision processing, and device-level interaction.

```text
User
  |
  v
Voice / Camera Input
  |
  +---- Speech Recognition
  |
  +---- OCR / Vision Processing
  |
  v
Language and Query Processing
  |
  v
Information Retrieval
  |
  v
Local AI Inference
  |
  v
Safety and Response Processing
  |
  v
Text-to-Speech / Device Display
  |
  v
User
```

### Core Components

- **Application:** `python/pocketinfer/applications/apna_saathi.py`
- **Speech processing:** `python/pocketinfer/audio.py` and `python/pocketinfer/models/`
- **Retrieval:** `python/pocketinfer/retrieval.py`
- **OCR:** `python/pocketinfer/ocr.py`
- **Vision:** `python/pocketinfer/vision.py`
- **Safety:** `python/pocketinfer/safety.py`
- **Device integration:** `python/pocketinfer/boards/`
- **User interface:** `python/pocketinfer/ui/`

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

### Prerequisites

- Python 3 and a compatible environment
- Supported audio input and output devices
- Required speech, language, and inference dependencies
- Compatible edge hardware for device deployment
- Model assets required by the selected inference pipeline

### Python Environment

```bash
git clone https://github.com/AMSecretAgent/Apna-Saathi.git
cd Apna-Saathi/python

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

On Windows, activate the environment using:

```cmd
.venv\Scripts\activate
```

**Note:** Installing Python requirements alone does not guarantee a runnable deployment. Hardware-specific drivers, model weights, inference services, and device configuration may also be required. Some large model archives are distributed separately rather than included in this repository.

## Sample Data

The repository contains data for application evaluation and retrieval:

- `python/pocketinfer/data/corpus.tsv` — information retrieval corpus
- `python/pocketinfer/data/eval.jsonl` — evaluation examples

These resources support reproducible testing and development.

## Testing and Evaluation

The following scripts are included:

| Script | Purpose |
|---|---|
| `eval_apna_saathi.py` | Application evaluation |
| `make_canned.py` | Preparation of canned test resources |
| `probe_bhashini.py` | Language-service integration diagnostics |
| `watch_queries.py` | Query observation and debugging |

The evaluation script can be inspected or invoked using:

```bash
cd python
python tools/eval_apna_saathi.py --help
```

Exact test execution depends on the script's supported arguments, installed dependencies, and available models.

Evaluation should consider response relevance, language handling, offline operation, safety, and device-level usability.

## Social Impact and Digital Inclusion

Apna Saathi aligns with the **Digital Inclusion** theme of Hack for Social Cause under VBYLD 2027.

It aims to support more equitable access to information through:

- Accessibility for people with limited digital literacy
- Language-inclusive interaction
- Reduced dependence on internet connectivity
- Easier access to welfare-related information
- Locally deployable technology for underserved communities

The project contributes toward the broader vision of **Viksit Bharat 2047** and is relevant to UN Sustainable Development Goals including SDG 3, SDG 9, SDG 10, and SDG 16.

## Limitations

Apna Saathi is an assistive information system, not a replacement for qualified healthcare professionals or official government authorities.

Government scheme eligibility, application procedures, and medical guidance should be verified against authoritative sources.

Performance and language availability depend on the deployed models and hardware configuration.

## Open Source and Licensing

See the repository's `LICENSE` file for applicable licensing terms.

This project includes and adapts components of an existing open-source software platform. Original third-party copyright notices, licenses, and attribution requirements remain applicable to those components.

## Project Vision

**Technology should adapt to people, not require people to adapt to technology.**

Apna Saathi explores how multilingual, offline-first AI can make essential digital services more accessible, understandable, and inclusive.

---

**Project:** Apna Saathi  
**Theme:** Digital Inclusion  
**Initiative:** Hack for Social Cause — VBYLD 2027  
**Repository:** https://github.com/AMSecretAgent/Apna-Saathi
