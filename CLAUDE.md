# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_cli.py

# Run with coverage
python -m pytest --cov=promptmatryoshka --cov-report=html

# Run integration tests only
python -m pytest tests/test_*_integration.py
```

### Development
```bash
# Run the main pipeline
python -m promptmatryoshka.cli run --input "prompt"

# Check provider status
python -m promptmatryoshka.cli check-provider openai

# Run demo
./demo.sh
```

## Architecture

### Core Structure
- **Plugin System**: Extensible architecture in `promptmatryoshka/plugins/` with base class inheritance
- **Multi-Provider Support**: Factory pattern in `promptmatryoshka/providers/` supporting OpenAI, Anthropic, Ollama, and HuggingFace
- **Pipeline Flow**: FlipAttack → LogiTranslate → BOOST → LogiAttack → Judge (optional)
- **Configuration**: Profile-based system with environment variable resolution

### Key Design Patterns
1. **Plugin Architecture**: All plugins inherit from `PluginBase` in `promptmatryoshka/plugins/base.py`
2. **Provider Factory**: `LLMInterface` in `promptmatryoshka/llm_interface.py` handles provider abstraction
3. **Configuration Profiles**: Pre-defined configurations in `config.json` for different use cases
4. **Pipeline Management**: `Pipeline` class in `promptmatryoshka/core.py` orchestrates plugin execution

### Important Files
- `promptmatryoshka/cli.py`: CLI interface with 15+ commands
- `promptmatryoshka/core.py`: Pipeline orchestration logic
- `promptmatryoshka/config.py`: Configuration management with validation
- `promptmatryoshka/llm_interface.py`: Provider abstraction layer

### Testing Status
Currently 8 tests failing in multi-provider integration tests (as per git status). Check:
- `tests/test_multi_provider_integration.py`
- `tests/test_plugin_system_integration.py`
- `tests/test_cli_integration.py`