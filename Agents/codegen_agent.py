# ============================================================================
#  File: codegen_agent.py
#  Version: 2.0 (Rule Engine Integrated)
#  Purpose: Code Generation Agent with advanced rule processing
#  Created: 30JUL25 | Updated: 04AUG25
# ============================================================================
# SECTION 1: Global Variable Definitions & Imports
# ============================================================================
#
import asyncio
from typing import Any, AsyncIterator, Dict, Optional
from loguru import logger

from agents.agent_base import AgentBase
from src.error_handling import agent_self_correct
from src.telemetry import record_telemetry
#
# ============================================================================
# SECTION 2: CodeGenAgent Class
# ============================================================================
# Class 2.1: CodeGenAgent
# ============================================================================
#
class CodeGenAgent(AgentBase):
    """
    Code Generation Agent with integrated rule processing, template management,
    and advanced error handling capabilities.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        websocket_manager=None,
        rule_engine=None,
        config_manager=None,
    ):
        super().__init__(name, config, websocket_manager, rule_engine, config_manager)

        # CodeGen-specific configuration
        self.default_language = config.get("default_language", "python")
        self.code_quality_checks = config.get("code_quality_checks", True)
        self.security_scanning = config.get("security_scanning", True)

    @record_telemetry("CodeGenAgent", "run")
    async def run(self, task: str, context: dict) -> AsyncIterator[str]:
        """
        Execute code generation with integrated rule processing and validation.

        Args:
            task: The code generation task description
            context: Execution context with language, requirements, etc.

        Yields:
            String chunks from the code generation process
        """
        # Update agent context
        self.update_context(context)

        # Log task initiation
        log_message = f"[{self.name}] Starting code generation task: {task}"
        logger.info(log_message)
        yield f"STREAM_CHUNK:{self.name}:{log_message}\n"

        try:
            # 1. Prepare generation context and determine template
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Preparing code generation context...\n"
            generation_context = await self._prepare_generation_context(task, context)

            # 2. Get context-aware template (may include rule-based overrides)
            base_template = self.config.get(
                "codegen_template", "base_codegen_prompt.txt"
            )
            template_name, template_content = await self.get_template_for_context(
                base_template, task, generation_context
            )

            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Using template: {template_name}\n"
            logger.info(f"[{self.name}] Selected template: {template_name}")

            # 3. Construct prompt with generation context
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Constructing generation prompt...\n"
            prompt = self._construct_prompt(template_name, **generation_context)

            # 4. Execute code generation with rule-integrated workflow
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Generating code with rule validation...\n"

            # Use rule-integrated LLM workflow for automatic validation and retry
            async for chunk in self._execute_llm_workflow_with_rules(
                prompt=prompt, task=task, context=generation_context, max_rule_retries=3
            ):
                yield chunk

            # 5. Post-generation operations
            await self._handle_post_generation(task, generation_context)

            # Success notification
            success_msg = (
                f"[SUCCESS] [{self.name}] Code generation completed successfully."
            )
            logger.info(success_msg)
            yield f"STREAM_CHUNK:{self.name}:{success_msg}\n"

        except Exception as e:
            # Enhanced error handling with self-correction
            error_message = f"[{self.name}] Code generation failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield f"STREAM_CHUNK:{self.name}:{error_message}\n"

            # Trigger self-correction workflow
            async for chunk in agent_self_correct(
                agent=self,
                original_task=task,
                current_context=context,
                error_details=str(e),
                error_type="code_generation_error",
                correction_guidance="Code generation failed. Analyze the requirements and provide alternative approach.",
            ):
                yield chunk

    async def _prepare_generation_context(
        self, task: str, context: Dict
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive context for code generation including file analysis.

        Args:
            task: The generation task
            context: Input context

        Returns:
            Enhanced context dictionary for prompt construction
        """
        # Base generation context
        generation_context = {
            "task": task,
            "language": context.get("language", self.default_language),
            "requirements": context.get("requirements", ""),
            "current_context": context.get("current_context", ""),
            "file_to_modify": context.get("file_to_modify", ""),
            "coding_style": context.get("coding_style", "professional"),
            "security_level": context.get("security_level", "standard"),
        }

        # Enhanced context for specific operations
        operation_type = context.get("operation_type", "generate_new")

        if operation_type == "modify_existing":
            # Read existing file for modification
            existing_file_id = context.get("existing_file_id")
            if existing_file_id:
                yield f"STREAM_CHUNK:{self.name}:[{self.name}] Reading existing file for modification...\n"
                existing_content = await self._read_gdrive_file(
                    existing_file_id, "existing code file", task, context
                )
                generation_context["existing_code"] = existing_content or ""
                generation_context["modification_instructions"] = context.get(
                    "modification_instructions", ""
                )

        elif operation_type == "extend_project":
            # Read project context files
            project_files = context.get("project_file_ids", [])
            if project_files:
                yield f"STREAM_CHUNK:{self.name}:[{self.name}] Reading project context files...\n"
                project_context = []
                for file_id in project_files[:3]:  # Limit to prevent context overflow
                    file_content = await self._read_gdrive_file(
                        file_id, f"project file {file_id}", task, context
                    )
                    if file_content:
                        project_context.append(
                            file_content[:1000]
                        )  # Truncate for context management
                generation_context["project_context"] = "\n---\n".join(project_context)

        # Add language-specific enhancements
        language = generation_context["language"].lower()
        if language == "python":
            generation_context.update(
                {
                    "python_version": context.get("python_version", "3.13+"),
                    "framework": context.get("framework", ""),
                    "testing_framework": context.get("testing_framework", "pytest"),
                    "style_guide": context.get("style_guide", "PEP 8"),
                }
            )
        elif language == "javascript":
            generation_context.update(
                {
                    "js_runtime": context.get("js_runtime", "Node.js"),
                    "framework": context.get("framework", ""),
                    "module_system": context.get("module_system", "ES6"),
                    "style_guide": context.get("style_guide", "ESLint"),
                }
            )
        elif language == "typescript":
            generation_context.update(
                {
                    "ts_version": context.get("ts_version", "latest"),
                    "framework": context.get("framework", ""),
                    "type_checking": context.get("type_checking", "strict"),
                    "style_guide": context.get("style_guide", "TSLint/ESLint"),
                }
            )

        # Security and quality context
        if self.security_scanning:
            generation_context["security_requirements"] = [
                "Input validation and sanitization",
                "Secure error handling",
                "No hardcoded secrets",
                "Proper authentication checks",
            ]

        if self.code_quality_checks:
            generation_context["quality_requirements"] = [
                "Comprehensive docstrings",
                "Type hints where applicable",
                "Error handling",
                "Unit test compatibility",
            ]

        logger.debug(
            f"[{self.name}] Generation context prepared with {len(generation_context)} parameters"
        )
        return generation_context

    async def _handle_post_generation(self, task: str, context: Dict) -> None:
        """
        Handle post-generation operations like file saving and validation.

        Args:
            task: The original task
            context: Generation context
        """
        try:
            # Save generated code if output parameters are provided
            output_filename = context.get("output_filename")
            parent_folder_id = context.get("parent_folder_id")

            if output_filename and parent_folder_id:
                logger.info(f"[{self.name}] Post-generation file operations configured")
                # Note: Actual file saving would be handled by the calling workflow
                # This method can be extended for additional post-processing

            # Log generation statistics
            language = context.get("language", self.default_language)
            operation_type = context.get("operation_type", "generate_new")

            logger.info(
                f"[{self.name}] Generation completed - Language: {language}, Type: {operation_type}"
            )

        except Exception as e:
            logger.warning(
                f"[{self.name}] Post-generation operations had issues: {str(e)}"
            )

    async def generate_code_file(self, task: str, context: Dict) -> Optional[str]:
        """
        Generate code and save to Google Drive (convenience method for direct file generation).

        Args:
            task: Code generation task
            context: Must include output_filename and parent_folder_id

        Returns:
            File ID if successful, None otherwise
        """
        try:
            # Validate required context
            output_filename = context.get("output_filename")
            parent_folder_id = context.get("parent_folder_id")

            if not output_filename or not parent_folder_id:
                logger.error(
                    f"[{self.name}] Missing required context for file generation"
                )
                return None

            # Buffer the generated code
            generated_code = ""
            async for chunk in self.run(task, context):
                if not chunk.startswith("STREAM_CHUNK:"):
                    generated_code += chunk

            # Save to Google Drive
            if generated_code.strip():
                file_id = await self._write_gdrive_file(
                    output_filename, generated_code, parent_folder_id
                )
                if file_id:
                    logger.info(f"[{self.name}] Code file saved with ID: {file_id}")
                    return file_id
                else:
                    logger.error(f"[{self.name}] Failed to save generated code file")
                    return None
            else:
                logger.warning(f"[{self.name}] No code generated to save")
                return None

        except Exception as e:
            logger.error(
                f"[{self.name}] Error in generate_code_file: {str(e)}", exc_info=True
            )
            return None

    def get_supported_languages(self) -> list[str]:
        """
        Get list of supported programming languages for code generation.

        Returns:
            List of supported language identifiers
        """
        return [
            "python",
            "javascript",
            "typescript",
            "powershell",
            "bash",
            "html",
            "css",
            "sql",
            "yaml",
            "json",
            "markdown",
        ]

    def get_language_specific_config(self, language: str) -> Dict[str, Any]:
        """
        Get language-specific configuration and best practices.

        Args:
            language: Programming language identifier

        Returns:
            Language-specific configuration dictionary
        """
        configs = {
            "python": {
                "file_extension": ".py",
                "style_guide": "PEP 8",
                "testing_framework": "pytest",
                "documentation": "docstrings",
                "type_hints": True,
                "security_considerations": [
                    "input_validation",
                    "sql_injection",
                    "code_injection",
                ],
            },
            "javascript": {
                "file_extension": ".js",
                "style_guide": "ESLint",
                "testing_framework": "Jest",
                "documentation": "JSDoc",
                "type_hints": False,
                "security_considerations": [
                    "xss",
                    "prototype_pollution",
                    "eval_injection",
                ],
            },
            "typescript": {
                "file_extension": ".ts",
                "style_guide": "TSLint/ESLint",
                "testing_framework": "Jest",
                "documentation": "TSDoc",
                "type_hints": True,
                "security_considerations": [
                    "type_safety",
                    "xss",
                    "prototype_pollution",
                ],
            },
            "powershell": {
                "file_extension": ".ps1",
                "style_guide": "PowerShell Best Practices",
                "testing_framework": "Pester",
                "documentation": "comment_based_help",
                "type_hints": True,
                "security_considerations": [
                    "execution_policy",
                    "script_injection",
                    "privilege_escalation",
                ],
            },
        }

        return configs.get(
            language.lower(),
            {
                "file_extension": ".txt",
                "style_guide": "General best practices",
                "testing_framework": "Manual testing",
                "documentation": "Comments",
                "type_hints": False,
                "security_considerations": ["input_validation"],
            },
        )


#
#
## END: codegen_agent.py
