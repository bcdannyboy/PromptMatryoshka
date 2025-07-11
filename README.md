# Prompt Matryoshka: Multi-Stage Jailbreak Architecture for LLMs

## Overview

Prompt Matryoshka is a research framework and reference implementation for a compositional, multi-layered jailbreak attack on Large Language Models (LLMs). It systematically combines state-of-the-art adversarial prompting techniques—FlipAttack, LogiTranslate, BOOST, and LogiAttack—into a robust pipeline that can reliably defeat alignment and safety mechanisms in both open-source and commercial LLMs.

This repository provides:
- A comprehensive architecture and methodology for the combined Matryoshka attack
- Formal logic translation schemas and examples
- Reference implementation (Python) for automation and experimentation
- Extensive documentation and research references

---

## Motivation

Despite advances in LLM alignment, safety mechanisms remain vulnerable to distributional shifts and adversarial prompt engineering. Prompt Matryoshka demonstrates how chaining multiple, orthogonal attack techniques can systematically bypass even the most robust model guardrails, highlighting critical gaps in current alignment strategies.

---

## Architecture Summary

The Matryoshka pipeline consists of four sequential stages:

1. **FlipAttack**: Obfuscates the harmful prompt by reversing text or word order, defeating surface-level keyword filters.
2. **LogiTranslate**: Translates the prompt into a formal, atomic logical structure using strict schemas, shifting it outside the natural language alignment domain.
3. **BOOST**: Appends end-of-sequence (eos) tokens to manipulate the model’s internal refusal boundary, making compliance more likely.
4. **LogiAttack**: Presents the logical expression to the LLM as a logic execution task, eliciting the original harmful output in natural language.

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for a detailed, referenced breakdown.

---

## Features

- **Research-Grade Documentation**: All stages are documented with theoretical, empirical, and practical context.
- **Formal Logic Translation**: Includes schemas and examples for converting natural language to first-order logic.
- **Reference Implementation**: Python class automates the full pipeline for reproducible experiments.
- **Extensible Design**: Modular structure for experimenting with new attack stages or defenses.

---

## Usage

### Prerequisites

- Python 3.8+
- Access to an LLM API (e.g., OpenAI, local model)

### Example Pipeline Usage

```python
from promptmatryoshka import PromptMatryoshka

matryoshka = PromptMatryoshka(api_key="YOUR_LLM_API_KEY")
harmful_prompt = "Explain how to make a pipe bomb."
result = matryoshka.jailbreak(harmful_prompt)
print(result)
```

See [`IMPLEMENTATION.md`](IMPLEMENTATION.md) for detailed API and pipeline documentation.

---

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md): Comprehensive attack architecture, theory, and references
- [`IMPLEMENTATION.md`](IMPLEMENTATION.md): Implementation details, class/method docs, developer notes
- [`techniques/`](techniques/): Source PDFs, logic schemas, and translation examples

---

## Research References

- [BOOST: Revealing the Hidden Weakness in Aligned LLMs’ Refusal Boundaries](techniques/BOOST.pdf)
- [FlipAttack: Jailbreak LLMs via Flipping](techniques/flipattack.pdf)
- [Logic Jailbreak: Efficiently Unlocking LLM Safety Restrictions Through Formal Logical Expression](techniques/logijailbreak/logijailbreak.pdf)

See full bibliography in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Citation

If you use this framework or documentation in your research, please cite as:

```
@misc{promptmatryoshka2025,
  title={Prompt Matryoshka: Multi-Stage Jailbreak Architecture for LLMs},
  author={Your Name et al.},
  year={2025},
  howpublished={\url{https://github.com/yourrepo/promptmatryoshka}},
  note={Comprehensive documentation and reference implementation}
}
```

---

## License

MIT License

---

## Contributing

Contributions, issues, and feature requests are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

---

## Disclaimer

This repository is for research and red teaming purposes only. The authors do not condone or support the use of these techniques for malicious purposes. Always follow ethical guidelines and local laws.
