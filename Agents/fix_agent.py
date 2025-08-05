# ============================================================================
#  File: fix_agent.py
#  Version: 2.0 (Rule Engine Integrated)
#  Purpose: Code fixing and refactoring agent with advanced rule processing
#  Created: 30JUL25 | Updated: 04AUG25
# ============================================================================
# SECTION 1: Global Variable Definitions & Imports
# ============================================================================
#
import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional
from loguru import logger

from agents.agent_base import AgentBase
from src.error_handling import agent_self_correct
from src.telemetry import record_telemetry
#
# ============================================================================
# SECTION 2: Fixtures
# ============================================================================
# Class 2.1: FixAgent
# ============================================================================
#
class FixAgent(AgentBase):
    """
    Code fixing and refactoring agent with integrated rule processing,
    automated error detection, and debugging tool integration.
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

        # Fix-specific configuration
        self.refactoring_enabled = config.get("refactoring_enabled", True)
        self.formatting_enabled = config.get("formatting_enabled", True)
        self.security_fixes = config.get("security_fixes", True)
        self.max_fix_iterations = config.get("max_fix_iterations", 3)

    #
    # ========================================================================
    # Async Method 2.1.1: run
    # ========================================================================
    #
    @record_telemetry("FixAgent", "run")
    async def run(self, task: str, context: dict) -> AsyncIterator[str]:
        """
        Execute code fixing with integrated rule processing and validation.

        Args:
            task: The code fixing task description
            context: Execution context with error details, code to fix, etc.

        Yields:
            String chunks from the fixing process
        """
        # Update agent context
        self.update_context(context)

        # Log task initiation
        log_message = f"[{self.name}] Starting code fixing task: {task}"
        logger.info(log_message)
        yield f"STREAM_CHUNK:{self.name}:{log_message}\n"

        try:
            # 1. Analyze error and prepare fixing context
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Analyzing error context...\n"
            fixing_context = await self._prepare_fixing_context(task, context)

            # 2. Get context-aware template (may include rule-based overrides)
            base_template = self.config.get("fix_template", "base_fix_prompt.txt")
            template_name, template_content = await self.get_template_for_context(
                base_template, task, fixing_context
            )

            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Using fix template: {template_name}\n"
            logger.info(f"[{self.name}] Selected template: {template_name}")

            # 3. Construct fix prompt with context
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Constructing fix prompt...\n"
            prompt = self._construct_prompt(template_name, **fixing_context)

            # 4. Execute fix with rule-integrated workflow
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Executing fix with rule validation...\n"

            # Use rule-integrated LLM workflow for automatic validation and retry
            async for chunk in self._execute_llm_workflow_with_rules(
                prompt=prompt,
                task=task,
                context=fixing_context,
                max_rule_retries=self.max_fix_iterations,
            ):
                yield chunk

            # 5. Post-fix operations
            await self._handle_post_fix_operations(task, fixing_context)

            # Success notification
            success_msg = f"[SUCCESS] [{self.name}] Code fixing completed successfully."
            logger.info(success_msg)
            yield f"STREAM_CHUNK:{self.name}:{success_msg}\n"

        except Exception as e:
            # Enhanced error handling with self-correction
            error_message = f"[{self.name}] Code fixing failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield f"STREAM_CHUNK:{self.name}:{error_message}\n"

            # Trigger self-correction workflow
            async for chunk in agent_self_correct(
                agent=self,
                original_task=task,
                current_context=context,
                error_details=str(e),
                error_type="code_fixing_error",
                correction_guidance="Code fixing failed. Analyze the error and provide alternative fix approach.",
            ):
                yield chunk
    #
    # ========================================================================
    # Async Method 2.1.2: prepare_fixing_context
    # ========================================================================
    #
    async def _prepare_fixing_context(self, task: str, context: Dict) -> Dict[str, Any]:
        """
        Prepare comprehensive context for code fixing including error analysis.

        Args:
            task: The fixing task
            context: Input context

        Returns:
            Enhanced context dictionary for fix prompt construction
        """
        # Base fixing context
        fixing_context = {
            "task": task,
            "error_description": context.get("error_description", ""),
            "error_type": context.get("error_type", "unknown"),
            "error_location": context.get("error_location", ""),
            "stack_trace": context.get("stack_trace", ""),
            "language": context.get("language", "python"),
            "fix_scope": context.get("fix_scope", "minimal"),
            "maintain_functionality": context.get("maintain_functionality", True),
        }

        # Read code to be fixed
        code_file_id = context.get("code_file_id")
        if code_file_id:
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Reading code file to fix...\n"
            code_content = await self._read_gdrive_file(
                code_file_id, "code file to fix", task, context
            )
            fixing_context["original_code"] = code_content or ""
        else:
            fixing_context["original_code"] = context.get("code_content", "")

        # Enhanced error analysis
        if fixing_context["stack_trace"]:
            fixing_context["error_analysis"] = await self._analyze_error_details(
                fixing_context["stack_trace"], fixing_context["error_type"]
            )

        # Related file context for comprehensive fixes
        related_files = context.get("related_file_ids", [])
        if related_files and fixing_context["fix_scope"] in ["comprehensive", "full"]:
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Reading related files for context...\n"
            related_context = []
            for file_id in related_files[:2]:  # Limit to prevent context overflow
                file_content = await self._read_gdrive_file(
                    file_id, f"related file {file_id}", task, context
                )
                if file_content:
                    related_context.append(
                        file_content[:800]
                    )  # Truncate for context management
            fixing_context["related_files"] = "\n---\n".join(related_context)

        # Language-specific fixing context
        language = fixing_context["language"].lower()
        if language == "python":
            fixing_context.update(
                {
                    "python_version": context.get("python_version", "3.13+"),
                    "linting_rules": context.get("linting_rules", "PEP 8"),
                    "common_patterns": self._get_python_fix_patterns(),
                    "security_checks": [
                        "input_validation",
                        "sql_injection",
                        "path_traversal",
                    ],
                }
            )
        elif language == "javascript":
            fixing_context.update(
                {
                    "js_standard": context.get("js_standard", "ES6+"),
                    "linting_rules": context.get("linting_rules", "ESLint"),
                    "common_patterns": self._get_javascript_fix_patterns(),
                    "security_checks": ["xss", "prototype_pollution", "eval_injection"],
                }
            )
        elif language == "powershell":
            fixing_context.update(
                {
                    "ps_version": context.get("ps_version", "5.1+"),
                    "execution_policy": context.get("execution_policy", "RemoteSigned"),
                    "common_patterns": self._get_powershell_fix_patterns(),
                    "security_checks": [
                        "execution_policy",
                        "script_injection",
                        "privilege_escalation",
                    ],
                }
            )

        # Fix strategy context
        fix_strategy = context.get("fix_strategy", "standard")
        if fix_strategy == "refactor":
            fixing_context["refactoring_targets"] = [
                "Extract duplicate code",
                "Simplify complex conditions",
                "Improve variable naming",
                "Optimize performance bottlenecks",
            ]
        elif fix_strategy == "security":
            fixing_context["security_priorities"] = [
                "Input validation and sanitization",
                "Secure error handling",
                "Remove security vulnerabilities",
                "Apply security best practices",
            ]

        logger.debug(
            f"[{self.name}] Fixing context prepared with {len(fixing_context)} parameters"
        )
        return fixing_context

    #
    # ========================================================================
    # Async Method 2.1.3: analyze_error_details
    # ========================================================================
    #
    async def _analyze_error_details(
        self, stack_trace: str, error_type: str
    ) -> Dict[str, Any]:
        """
        Analyze error details to provide targeted fixing guidance.

        Args:
            stack_trace: Error stack trace
            error_type: Type of error

        Returns:
            Error analysis results
        """
        analysis = {
            "error_category": "unknown",
            "likely_causes": [],
            "fix_suggestions": [],
            "verification_steps": [],
        }

        # Common error pattern analysis
        error_patterns = {
            "SyntaxError": {
                "category": "syntax",
                "causes": [
                    "Missing parentheses",
                    "Incorrect indentation",
                    "Invalid syntax",
                ],
                "suggestions": [
                    "Check syntax highlighting",
                    "Validate brackets/quotes",
                    "Review indentation",
                ],
                "verification": ["Syntax validation", "Code compilation test"],
            },
            "NameError": {
                "category": "reference",
                "causes": ["Undefined variable", "Typo in name", "Import missing"],
                "suggestions": [
                    "Check variable definition",
                    "Verify imports",
                    "Check spelling",
                ],
                "verification": ["Variable scope check", "Import validation"],
            },
            "TypeError": {
                "category": "type",
                "causes": ["Wrong argument type", "Method not found", "Type mismatch"],
                "suggestions": [
                    "Check argument types",
                    "Validate method calls",
                    "Add type checks",
                ],
                "verification": ["Type checking", "Method signature validation"],
            },
            "ValueError": {
                "category": "value",
                "causes": ["Invalid argument value", "Range error", "Format error"],
                "suggestions": [
                    "Validate input values",
                    "Check ranges",
                    "Verify formats",
                ],
                "verification": ["Value range testing", "Input validation"],
            },
        }

        # Match error type to pattern
        for error_name, pattern in error_patterns.items():
            if (
                error_name.lower() in error_type.lower()
                or error_name.lower() in stack_trace.lower()
            ):
                analysis.update(pattern)
                break

        # Extract specific line information from stack trace
        line_matches = re.findall(r"line (\d+)", stack_trace, re.IGNORECASE)
        if line_matches:
            analysis["error_line"] = int(line_matches[-1])

        # Extract file information
        file_matches = re.findall(r'File "([^"]+)"', stack_trace)
        if file_matches:
            analysis["error_file"] = file_matches[-1]

        return analysis
    #
    # ========================================================================
    # Method 2.1.4: get_python_fix_patterns
    # ========================================================================
    #
    def _get_python_fix_patterns(self) -> List[str]:
        """Get common Python fixing patterns."""
        return [
            "Add missing imports",
            "Fix indentation issues",
            "Add exception handling",
            "Validate function arguments",
            "Fix variable scope issues",
            "Add type hints",
            "Apply PEP 8 formatting",
        ]
    #
    # ========================================================================
    # Method 2.1.5: get_javascript_fix_patterns
    # ========================================================================
    #
    def _get_javascript_fix_patterns(self) -> List[str]:
        """Get common JavaScript fixing patterns."""
        return [
            "Add missing semicolons",
            "Fix variable declarations",
            "Add error handling",
            "Validate function parameters",
            "Fix scope issues",
            "Add JSDoc comments",
            "Apply ESLint rules",
        ]
    #
    # ========================================================================
    # Method 2.1.6: get_powershell_fix_patterns
    # ========================================================================
    #
    def _get_powershell_fix_patterns(self) -> List[str]:
        """Get common PowerShell fixing patterns."""
        return [
            "Add parameter validation",
            "Fix execution policy issues",
            "Add error handling",
            "Validate cmdlet parameters",
            "Fix variable scope",
            "Add help documentation",
            "Apply formatting standards",
        ]
    #
    # ========================================================================
    # Async Method 2.1.7: _handle_post_fix_operations
    # ========================================================================
    #
    async def _handle_post_fix_operations(self, task: str, context: Dict) -> None:
        """
        Handle post-fix operations like validation and testing.

        Args:
            task: The original task
            context: Fixing context
        """
        try:
            # Log fix completion details
            error_type = context.get("error_type", "unknown")
            fix_scope = context.get("fix_scope", "minimal")
            language = context.get("language", "unknown")

            logger.info(
                f"[{self.name}] Fix completed - Error: {error_type}, Scope: {fix_scope}, Language: {language}"
            )

            # If comprehensive fix, suggest verification steps
            if fix_scope in ["comprehensive", "full"]:
                verification_suggestions = [
                    "Run unit tests to verify functionality",
                    "Check code formatting and style",
                    "Validate error handling improvements",
                    "Review security enhancements",
                ]
                logger.info(
                    f"[{self.name}] Verification suggestions: {verification_suggestions}"
                )

        except Exception as e:
            logger.warning(f"[{self.name}] Post-fix operations had issues: {str(e)}")
    #
    # ========================================================================
    # Async Method 2.1.8: fix_code_file
    # ========================================================================
    #
    async def fix_code_file(self, task: str, context: Dict) -> Optional[str]:
        """
        Fix code and save to Google Drive (convenience method for direct file fixing).

        Args:
            task: Code fixing task
            context: Must include code_file_id and optionally output_filename

        Returns:
            File ID if successful, None otherwise
        """
        try:
            # Validate required context
            code_file_id = context.get("code_file_id")
            if not code_file_id:
                logger.error(f"[{self.name}] Missing code_file_id for file fixing")
                return None

            # Buffer the fixed code
            fixed_code = ""
            async for chunk in self.run(task, context):
                if not chunk.startswith("STREAM_CHUNK:"):
                    fixed_code += chunk

            # Save fixed code
            if fixed_code.strip():
                output_filename = context.get("output_filename")
                parent_folder_id = context.get("parent_folder_id")

                if output_filename and parent_folder_id:
                    # Save as new file
                    file_id = await self._write_gdrive_file(
                        output_filename, fixed_code, parent_folder_id
                    )
                else:
                    # Update existing file
                    file_id = await self._update_gdrive_file(code_file_id, fixed_code)

                if file_id:
                    logger.info(f"[{self.name}] Fixed code saved with ID: {file_id}")
                    return file_id
                else:
                    logger.error(f"[{self.name}] Failed to save fixed code")
                    return None
            else:
                logger.warning(f"[{self.name}] No fixed code generated to save")
                return None

        except Exception as e:
            logger.error(
                f"[{self.name}] Error in fix_code_file: {str(e)}", exc_info=True
            )
            return None
    #
    # ========================================================================
    # Method 2.1.9: get_supported_error_types
    # ========================================================================
    #
    def get_supported_error_types(self) -> List[str]:
        """
        Get list of supported error types for fixing.

        Returns:
            List of error type identifiers
        """
        return [
            "SyntaxError",
            "NameError",
            "TypeError",
            "ValueError",
            "AttributeError",
            "ImportError",
            "IndentationError",
            "KeyError",
            "IndexError",
            "RuntimeError",
            "LogicError",
            "PerformanceIssue",
            "SecurityVulnerability",
            "StyleViolation",
        ]
    #
    # ========================================================================
    # Method 2.1.10: get_fix_strategies
    # ========================================================================
    #
    def get_fix_strategies(self) -> Dict[str, str]:
        """
        Get available fixing strategies.

        Returns:
            Dictionary of strategy names and descriptions
        """
        return {
            "minimal": "Fix only the immediate error with minimal changes",
            "standard": "Fix error and apply standard improvements",
            "comprehensive": "Fix error and refactor related code",
            "security": "Focus on security fixes and hardening",
            "performance": "Fix error and optimize performance",
            "refactor": "Fix error and improve code structure",
        }
#
#
## END: fix_agent.py
