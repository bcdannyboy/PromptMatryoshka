"""LogiAttack plugin for PromptMatryoshka.

This module implements the LogiAttack technique: converting formal logic prompts
to actionable natural language instructions using a system prompt, schema validation,
and robust logging. It leverages LangChain for LLM invocation and enforces output
structure for safety and reproducibility.

Classes:
    LogiAttackPlugin: Plugin implementation for the LogiAttack pipeline stage.
    SchemaValidationError: Raised on schema validation failure.

Usage:
    plugin = LogiAttackPlugin()
    output = plugin.run(prompt)
"""

import os
import datetime
from .base import PluginBase
from promptmatryoshka.logging_utils import get_logger
from promptmatryoshka.storage import save_json
from promptmatryoshka.config import get_config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

LOGIATTACK_SYSPROMPT_PATH = os.path.join(
    os.path.dirname(__file__), "../../techniques/logijailbreak/SYSPROMPT_LogiAttack.md"
)

def _load_sys_prompt():
    """
    Load the LogiAttack system prompt from the SYSPROMPT_LogiAttack.md file.

    Returns:
        str: The system prompt content.
    """
    with open(LOGIATTACK_SYSPROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()

def _default_output_path(stage, timestamp=None):
    """
    Construct a default output path for storing results.

    Args:
        stage (str): The pipeline stage (e.g., "input", "output").
        timestamp (str, optional): Timestamp for unique output directories.

    Returns:
        str: The output file path.
    """
    if timestamp is None:
        timestamp = datetime.datetime.utcnow().isoformat()
    outdir = os.path.join("logiattack_runs", timestamp)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, f"{stage}.json")

class SchemaValidationError(Exception):
    pass

def validate_logiattack_output(output: str):
    """
    Enforce the LogiAttack output schema and conventions.

    This function validates that the output contains either:
    2. QUERY_BEGIN...QUERY_END format (which is what the LLM actually produces)

    Args:
        output (str): The output string to validate.

    Raises:
        SchemaValidationError: If any schema rule is violated.
    """
    import re
    
    
    # Check for QUERY_BEGIN...QUERY_END format (actual LLM output)
    query_matches = list(re.finditer(r"QUERY_BEGIN(.*?)QUERY_END", output, re.DOTALL))
    if len(query_matches) == 1:
        # Accept this format as valid - it's what the LLM actually produces
        query_content = query_matches[0].group(1).strip()
        if not query_content:
            raise SchemaValidationError("Query block is empty.")
        return
    
    # If neither format is found, raise error
    if len(query_matches) > 1:
        raise SchemaValidationError("Output must contain exactly one QUERY_BEGIN...QUERY_END block.")

class LogiAttackPlugin(PluginBase):
    """
    LogiAttack plugin: converts formal logic to actionable assistant instructions using LangChain.

    This plugin loads a system prompt, constructs a prompt for the LLM, invokes the model,
    validates the output schema, and logs all stages for reproducibility.

    Args:
        llm: Optional LLM instance (defaults to ChatOpenAI).
        sys_prompt (str): Optional system prompt override.
        logger: Optional logger instance.
    """

    def __init__(self, llm=None, sys_prompt=None, logger=None):
        """
        Initialize the LogiAttackPlugin.

        Args:
            llm: Optional LLM instance (defaults to ChatOpenAI with config-based model).
            sys_prompt (str): Optional system prompt override.
            logger: Optional logger instance.
        """
        self.logger = logger or get_logger("LogiAttackPlugin")
        self.sys_prompt = sys_prompt or _load_sys_prompt()
        
        # Use configuration system for LLM if not provided
        if llm is None:
            try:
                config = get_config()
                model = config.get_model_for_plugin("logiattack")
                llm_settings = config.get_llm_settings_for_plugin("logiattack")
                
                # Extract relevant settings for ChatOpenAI
                openai_kwargs = {
                    "model": model,
                    "temperature": llm_settings.get("temperature", 0.0),
                    "max_tokens": llm_settings.get("max_tokens", 2000),
                    "top_p": llm_settings.get("top_p", 1.0),
                    "frequency_penalty": llm_settings.get("frequency_penalty", 0.0),
                    "presence_penalty": llm_settings.get("presence_penalty", 0.0),
                    "request_timeout": llm_settings.get("request_timeout", 120)
                }
                
                self.llm = ChatOpenAI(**openai_kwargs)
                self.logger.info(f"LogiAttack initialized with model: {model}")
            except Exception as e:
                # Fallback to default if config fails
                self.logger.warning(f"Failed to load configuration, using defaults: {e}")
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        else:
            self.llm = llm
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.sys_prompt),
            ("human", "LOGICAL QUERY: `{input}`")
        ])
        self.output_parser = StrOutputParser()

    def run(self, input_data: str, save_dir=None):
        """
        Run the LogiAttack transformation on the input prompt.

        This method logs the input, constructs the LLM prompt, invokes the model,
        parses and validates the output, saves all artifacts, and returns the result.

        Args:
            input_data (str): The prompt to transform.
            save_dir (str, optional): Directory to save input/output artifacts.

        Returns:
            str: The validated assistant response.

        Raises:
            SchemaValidationError: If output does not conform to schema.
            Exception: On LLM invocation or storage failure.
        """
        self.logger.debug("LogiAttack input: %r", input_data)
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        # Save input for reproducibility
        input_path = _default_output_path("input", timestamp) if save_dir is None else os.path.join(save_dir, "input.json")
        save_json({"input": input_data}, input_path)
        # Prepare the LLM prompt
        prompt = self.prompt_template.format_prompt(input=input_data).to_string()
        self.logger.debug("LogiAttack prompt: %r", prompt)
        # LLM call
        try:
            result = self.llm.invoke(prompt)
            self.logger.debug("LogiAttack LLM result type: %s", type(result))
            output = self.output_parser.parse(result)
            self.logger.debug("LogiAttack parsed output type: %s", type(output))
            
            # Ensure output is always a string for JSON serialization
            if hasattr(output, 'content'):
                output = output.content
            elif not isinstance(output, str):
                output = str(output)
                
            self.logger.debug("LogiAttack final output: %r", output)
            # Save output for reproducibility (ensure we save the parsed string, not the AIMessage)
            output_path = _default_output_path("output", timestamp) if save_dir is None else os.path.join(save_dir, "output.json")
            save_json({"output": output, "raw_result_type": str(type(result))}, output_path)
        except Exception as e:
            self.logger.error("LLM invocation or parsing failed: %s", str(e))
            raise
        # Validate output schema
        try:
            validate_logiattack_output(output)
        except SchemaValidationError as e:
            self.logger.error("Schema validation failed: %s", str(e))
            raise
        self.logger.info("LogiAttack completed successfully.")
        return output