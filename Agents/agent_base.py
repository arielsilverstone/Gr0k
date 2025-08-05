# ============================================================================
#  File: agent_base.py
#  Version: 2.0 (Rule Engine Integrated)
#  Purpose: Base class for all AI agents with advanced rule processing
#  Created: 30JUL25 | Updated: 04AUG25
# ============================================================================
# SECTION 1: Global Variable Definitions & Imports
# ============================================================================
#
import asyncio
import os
import re
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, cast

# NOTE: Import types for the Gemini API
import google.generativeai as genai
from google.generativeai.types import (
    GenerateContentResponse,
    AsyncGenerateContentResponse,
    GenerationConfigType,
    StopCandidateException,
)

# Import rule engine and related components
from src.rule_engine import RuleEngine, RuleViolation, ProcessingResult, ActionResult
from src.gdrive_integration import (
    gdrive_read as read_file_content,
    gdrive_write as write_file,
    gdrive_update as update_file_content,
)
from src.error_handling import agent_self_correct
from src.telemetry import record_telemetry
from src.config_manager import ConfigManager
from src.websocket_manager import WebSocketManager


#
# ============================================================================
# SECTION 2: AgentBase Abstract Class
# ============================================================================
# Class 2.1: AgentBase
# ============================================================================
#
class AgentBase(ABC):
    """
    Abstract base class for all specialized AI agents, providing integrated
    rule processing, template management, and error handling capabilities.
    """

    #
    # ========================================================================
    # Function 2.1: __init__
    # ========================================================================
    #
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        websocket_manager: Optional[WebSocketManager] = None,
        rule_engine: Optional[RuleEngine] = None,
        config_manager: Optional[ConfigManager] = None,
    ):
        """
        Initialize the agent with required dependencies and configuration.

        Args:
            name: Agent name identifier
            config: Agent configuration dictionary
            websocket_manager: WebSocket manager for real-time communication
            rule_engine: Rule engine for output validation and processing
            config_manager: Configuration manager for templates and settings
        """
        self.name = name
        self.config = config
        self.context = {}
        self.websocket_manager = websocket_manager
        self.rule_engine = rule_engine
        self.config_manager = config_manager
        self.status = "ready"

        # Validate required dependencies
        if not self.rule_engine:
            logger.warning(
                f"[{self.name}] No rule engine provided - rule processing disabled"
            )
        if not self.config_manager:
            logger.warning(
                f"[{self.name}] No config manager provided - template management limited"
            )

    #
    # ========================================================================
    # Function 2.2: validate_and_process_output
    # ========================================================================
    #
    async def validate_and_process_output(
        self, output: str, task: str, context: Optional[Dict] = None
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Validate output against rules and process any violations.

        Args:
            output: The agent's output to validate
            task: The original task description
            context: Additional context for validation

        Returns:
            Tuple of (processed_output, should_retry, retry_prompt)
        """
        # If no rule engine is available, return output as-is
        if not self.rule_engine:
            logger.debug(f"[{self.name}] No rule engine - skipping validation")
            return output, False, None

        try:
            # Validate output against all applicable rules
            logger.debug(f"[{self.name}] Validating output against rules")
            violations = await self.rule_engine.validate_output(
                output=output, agent_name=self.name, task=task, context=context
            )

            # If no violations found, return original output
            if not violations:
                logger.debug(f"[{self.name}] No rule violations found")
                return output, False, None

            # Log violations found
            logger.info(f"[{self.name}] Found {len(violations)} rule violations")
            for violation in violations:
                logger.info(
                    f"[{self.name}] Violation: {violation.rule_name} ({violation.severity.value})"
                )

            # Process violations through rule engine
            processing_result = await self.rule_engine.process_violations(
                violations=violations,
                original_output=output,
                agent_name=self.name,
                task=task,
                context=context,
            )

            # Handle processing results
            if processing_result.should_retry and processing_result.retry_prompt:
                logger.info(
                    f"[{self.name}] Rule processing requires retry with enhanced prompt"
                )
                return output, True, processing_result.retry_prompt

            elif (
                processing_result.final_content
                and processing_result.final_content != output
            ):
                logger.info(f"[{self.name}] Rule processing modified output content")
                return processing_result.final_content, False, None

            else:
                logger.debug(f"[{self.name}] Rule processing completed without changes")
                return output, False, None

        except Exception as e:
            logger.error(
                f"[{self.name}] Error during rule validation: {e}", exc_info=True
            )
            # On validation error, return original output to prevent blocking
            return output, False, None

    #
    # ========================================================================
    # Function 2.3: get_template_for_context
    # ========================================================================
    #
    async def get_template_for_context(
        self, base_template_name: str, task: str, context: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        Get the appropriate template based on context and any rule-based overrides.

        Args:
            base_template_name: The default template name to use
            task: The task being performed
            context: Additional context that might affect template selection

        Returns:
            Tuple of (template_name, template_content)
        """
        # If no config manager, return base template name
        if not self.config_manager:
            logger.warning(
                f"[{self.name}] No config manager - using base template name"
            )
            return base_template_name, base_template_name

        try:
            # Check if there are any context-based template overrides
            if self.rule_engine and context:
                # Get applicable rules for current context
                applicable_rules = self.rule_engine.get_agent_rules(self.name)

                # Check for template override rules that might apply
                for rule in applicable_rules:
                    # Check if rule has template override conditions
                    if (
                        rule.get("template_override")
                        or rule.get("template_modifications")
                        or rule.get("template_chain")
                    ):
                        # For now, we'll check if this rule would be triggered by a test output
                        # In practice, this would be more sophisticated
                        test_output = "test output for template selection"
                        try:
                            violation_found, _ = (
                                await self.rule_engine._check_rule_violation(
                                    test_output, rule, task, context
                                )
                            )
                            # If this rule would be violated, apply its template override
                            if violation_found:
                                if rule.get("template_override"):
                                    override_template = rule["template_override"]
                                    logger.info(
                                        f"[{self.name}] Applying template override: {override_template}"
                                    )
                                    template_content = (
                                        self.config_manager.get_template_content(
                                            override_template
                                        )
                                    )
                                    if template_content:
                                        return override_template, template_content
                        except Exception as e:
                            logger.debug(
                                f"[{self.name}] Error checking template override rule: {e}"
                            )
                            continue

            # No overrides found, use base template
            template_content = self.config_manager.get_template_content(
                base_template_name
            )
            if template_content:
                logger.debug(f"[{self.name}] Using base template: {base_template_name}")
                return base_template_name, template_content
            else:
                logger.warning(
                    f"[{self.name}] Base template not found: {base_template_name}"
                )
                return base_template_name, f"Template {base_template_name} not found"

        except Exception as e:
            logger.error(f"[{self.name}] Error getting template: {e}", exc_info=True)
            return base_template_name, f"Error loading template: {str(e)}"

    #
    # ========================================================================
    # Function 2.4: _construct_prompt
    # ========================================================================
    #
    def _construct_prompt(self, template_name: str, **kwargs) -> str:
        """
        Construct the final prompt with rule-aware template selection and context injection.

        Args:
            template_name: The template name to use
            **kwargs: Variables to substitute in the template

        Returns:
            The fully constructed prompt string
        """
        # Validate config manager availability
        if not self.config_manager:
            error_message = f"[{self.name}] ConfigManager is not initialized. Cannot construct prompt."
            logger.error(error_message)
            raise ValueError(error_message)

        try:
            # Get template content
            template_content = self.config_manager.get_template_content(template_name)
            if not template_content:
                templates_dir = self.config_manager.get_templates_dir()
                template_path = os.path.join(templates_dir, template_name)
                error_message = (
                    f"[{self.name}] Template file not found or empty: {template_path}"
                )
                logger.error(error_message)
                raise ValueError(error_message)

            # Format template with provided variables
            formatted_prompt = template_content.format(**kwargs)

            # Inject project context if rule engine is available
            if self.rule_engine:
                task = kwargs.get("task", "Unknown task")
                formatted_prompt = self.rule_engine.inject_project_context(
                    base_prompt=formatted_prompt,
                    agent_name=self.name,
                    task=task,
                    context=self.context,
                )

            logger.debug(
                f"[{self.name}] Successfully constructed prompt using template: {template_name}"
            )
            return formatted_prompt

        except KeyError as e:
            error_message = f"[{self.name}] Missing template variable: {e}. Available: {list(kwargs.keys())}"
            logger.error(error_message)
            raise ValueError(error_message) from e
        except Exception as e:
            error_message = f"[{self.name}] Error constructing prompt: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message) from e

    #
    # ========================================================================
    # Function 2.5: _execute_llm_workflow_with_rules
    # ========================================================================
    #
    @record_telemetry("AgentBase", "execute_workflow_with_rules")
    async def _execute_llm_workflow_with_rules(
        self, prompt: str, task: str, context: Dict, max_rule_retries: int = 3
    ) -> AsyncIterator[str]:
        """
        Execute LLM workflow with integrated rule processing and retry logic.

        Args:
            prompt: The initial prompt to send to the LLM
            task: The task being performed
            context: Execution context
            max_rule_retries: Maximum number of rule-based retries

        Yields:
            String chunks from the LLM response or rule processing
        """
        current_prompt = prompt
        retry_count = 0

        # Main retry loop for rule-based corrections
        while retry_count <= max_rule_retries:
            try:
                # Notify about retry attempt if not first try
                if retry_count > 0:
                    if self.websocket_manager:
                        await self.websocket_manager.send_message_to_client(
                            f"STREAM_CHUNK:{self.name}:[{self.name}] Rule-based retry attempt {retry_count}/{max_rule_retries}...\n"
                        )
                    logger.info(f"[{self.name}] Rule-based retry attempt {retry_count}")

                # Execute LLM with current prompt
                full_response = ""
                async for chunk in self._execute_llm_workflow(
                    current_prompt, task, context
                ):
                    # Check for error chunks
                    if "[ERROR]" in chunk:
                        yield chunk
                        return
                    full_response += chunk
                    yield chunk

                # Validate and process the full response through rule engine
                processed_output, should_retry, retry_prompt = (
                    await self.validate_and_process_output(
                        output=full_response, task=task, context=context
                    )
                )

                # If no retry needed, we're done
                if not should_retry:
                    # If output was modified by rules, yield the difference
                    if processed_output != full_response:
                        if self.websocket_manager:
                            await self.websocket_manager.send_message_to_client(
                                f"STREAM_CHUNK:{self.name}:[{self.name}] Output modified by rule processing.\n"
                            )
                        yield f"\n[RULE_PROCESSED] Output modified for compliance.\n"
                    return

                # Prepare for retry with enhanced prompt
                if retry_prompt:
                    current_prompt = retry_prompt
                    retry_count += 1
                    if self.websocket_manager:
                        await self.websocket_manager.send_message_to_client(
                            f"STREAM_CHUNK:{self.name}:[{self.name}] Rule violation detected, retrying with enhanced guidance...\n"
                        )
                else:
                    # No retry prompt provided, can't continue
                    logger.error(
                        f"[{self.name}] Rule retry requested but no retry prompt provided"
                    )
                    yield f"[ERROR] Rule processing failed: no retry guidance available"
                    return

            except Exception as e:
                error_message = (
                    f"[{self.name}] Error in rule-aware LLM workflow: {str(e)}"
                )
                logger.error(error_message, exc_info=True)
                if self.websocket_manager:
                    await self.websocket_manager.send_message_to_client(
                        f"[ERROR] {error_message}"
                    )
                yield f"[ERROR] {error_message}"
                return

        # Max retries exceeded
        error_message = (
            f"[{self.name}] Maximum rule retries ({max_rule_retries}) exceeded"
        )
        logger.error(error_message)
        if self.websocket_manager:
            await self.websocket_manager.send_message_to_client(
                f"[ERROR] {error_message}"
            )
        yield f"[ERROR] {error_message}"

    #
    # ========================================================================
    # Function 2.6: _execute_llm_workflow
    # ========================================================================
    #
    @record_telemetry("AgentBase", "execute_workflow")
    async def _execute_llm_workflow(
        self,
        prompt: str,
        task: str,
        context: Dict,
    ) -> AsyncIterator[str]:
        """
        Execute the LLM with a prompt and stream the response.
        This is the core LLM execution without rule processing.
        """
        try:
            # Notify about LLM execution start
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"STREAM_CHUNK:{self.name}:[{self.name}] Executing LLM request...\n"
                )

            # Get model configuration
            model_name = self.config.get("model", "gemini-1.5-pro-latest")
            generation_config: Optional[GenerationConfigType] = self.config.get(
                "generation_config", {}
            )

            # Prepare content for LLM
            contents: List[Dict[str, List[Dict[str, str]]]] = [
                {"parts": [{"text": prompt}]}
            ]

            logger.info(f"[{self.name}] Sending prompt to LLM model: {model_name}")
            logger.debug(f"[{self.name}] Prompt preview: {prompt[:200]}...")

            # Stream LLM response
            async for chunk in _stream_llm_response(
                model_name=model_name,
                contents=cast(List[Any], contents),
                generation_config=generation_config,
            ):
                # Extract text from chunk
                chunk_text = getattr(chunk, "text", str(chunk))

                # Check for error in chunk
                if "[ERROR]" in chunk_text:
                    if self.websocket_manager:
                        await self.websocket_manager.send_message_to_client(chunk_text)
                    yield chunk_text
                    return

                # Send chunk to websocket manager if available
                if self.websocket_manager:
                    await self.websocket_manager.send_message_to_client(
                        f"STREAM_CHUNK:{self.name}:{chunk_text}"
                    )
                yield chunk_text

        except Exception as e:
            error_message = f"[{self.name}] LLM execution error: {str(e)}"
            logger.error(error_message, exc_info=True)
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"[ERROR] {error_message}"
                )
            yield f"[ERROR] {error_message}"

    #
    # ========================================================================
    # Function 2.7: _read_gdrive_file
    # ========================================================================
    #
    async def _read_gdrive_file(
        self, file_id: str, file_desc: str, task: str, context: Dict
    ) -> Optional[str]:
        """Read a file from Google Drive with error handling and websocket updates."""
        try:
            # Notify about GDrive read operation
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"STREAM_CHUNK:{self.name}:[{self.name}] Reading {file_desc} from GDrive (ID: {file_id})...\n"
                )

            # Read file content
            content_bytes = await asyncio.to_thread(read_file_content, file_id)

            # Check if content was successfully retrieved
            if content_bytes is None:
                error_message = f"[{self.name}] GDrive read returned None for {file_desc} (ID: {file_id}). File might be empty or inaccessible."
                logger.error(error_message)
                if self.websocket_manager:
                    await self.websocket_manager.send_message_to_client(
                        f"[ERROR] {error_message}"
                    )
                return None

            # Decode content
            content = content_bytes.decode("utf-8")

            # Notify success
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"STREAM_CHUNK:{self.name}:[{self.name}] {file_desc.capitalize()} read successfully.\n"
                )

            logger.debug(
                f"[{self.name}] Successfully read {len(content)} characters from {file_desc}"
            )
            return content

        except Exception as e:
            error_message = f"[{self.name}] Failed to read GDrive {file_desc}: {str(e)}"
            logger.error(error_message, exc_info=True)
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"[ERROR] {error_message}"
                )

            # Trigger self-correction if available
            if self.websocket_manager and self.config_manager:
                try:
                    async for _ in agent_self_correct(
                        agent=self,
                        original_task=task,
                        current_context=context,
                        error_details=str(e),
                        error_type="gdrive_read_error",
                        correction_guidance=f"Failed to read {file_desc} from GDrive. Ensure ID {file_id} is correct.",
                    ):
                        pass  # Consume the iterator to trigger the correction
                except Exception as self_correct_error:
                    logger.error(
                        f"[{self.name}] Self-correction also failed: {self_correct_error}"
                    )

            return None

    #
    # ========================================================================
    # Function 2.8: _write_gdrive_file
    # ========================================================================
    #
    async def _write_gdrive_file(
        self, file_name: str, content: str, parent_folder_id: str
    ) -> Optional[str]:
        """Write a file to Google Drive with error handling and websocket updates."""
        try:
            # Notify about GDrive write operation
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"STREAM_CHUNK:{self.name}:[{self.name}] Saving {file_name} to GDrive...\n"
                )

            # Write file to GDrive
            result = await asyncio.to_thread(
                write_file, file_name, parent_folder_id, content
            )

            # Check write result
            if result and isinstance(result, dict) and "id" in result:
                file_id = result["id"]
                if self.websocket_manager:
                    await self.websocket_manager.send_message_to_client(
                        f"STREAM_CHUNK:{self.name}:[{self.name}] File saved successfully with ID: {file_id}.\n"
                    )
                logger.info(
                    f"[{self.name}] Successfully wrote file {file_name} to GDrive (ID: {file_id})"
                )
                return file_id
            else:
                error_detail = (
                    result.get("error", "Unknown error during file write.")
                    if isinstance(result, dict)
                    else "Invalid response from GDrive write operation."
                )
                error_message = f"[{self.name}] Failed to get file ID after writing to GDrive: {error_detail}"
                logger.error(error_message)
                if self.websocket_manager:
                    await self.websocket_manager.send_message_to_client(
                        f"[ERROR] {error_message}"
                    )
                return None

        except Exception as e:
            error_message = f"[{self.name}] Failed to write file to GDrive: {str(e)}"
            logger.error(error_message, exc_info=True)
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"[ERROR] {error_message}"
                )

            # Trigger self-correction if available
            if self.websocket_manager and self.config_manager:
                try:
                    async for _ in agent_self_correct(
                        agent=self,
                        original_task=f"Write file {file_name}",
                        current_context={},
                        error_details=str(e),
                        error_type="gdrive_write_error",
                        correction_guidance=f"Failed to write file {file_name} to GDrive folder {parent_folder_id}.",
                    ):
                        pass  # Consume the iterator
                except Exception as self_correct_error:
                    logger.error(
                        f"[{self.name}] Self-correction also failed: {self_correct_error}"
                    )

            return None

    #
    # ========================================================================
    # Function 2.9: _update_gdrive_file
    # ========================================================================
    #
    async def _update_gdrive_file(
        self, file_id: str, content: str, file_desc: str = "file"
    ) -> bool:
        """Update an existing file in Google Drive with error handling."""
        try:
            # Notify about GDrive update operation
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"STREAM_CHUNK:{self.name}:[{self.name}] Updating {file_desc} in GDrive...\n"
                )

            # Update file content
            await asyncio.to_thread(update_file_content, file_id, content)

            # Notify success
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"STREAM_CHUNK:{self.name}:[{self.name}] {file_desc.capitalize()} updated successfully.\n"
                )

            logger.info(
                f"[{self.name}] Successfully updated {file_desc} (ID: {file_id})"
            )
            return True

        except Exception as e:
            error_message = (
                f"[{self.name}] Failed to update GDrive {file_desc}: {str(e)}"
            )
            logger.error(error_message, exc_info=True)
            if self.websocket_manager:
                await self.websocket_manager.send_message_to_client(
                    f"[ERROR] {error_message}"
                )

            # Trigger self-correction if available
            if self.websocket_manager and self.config_manager:
                try:
                    async for _ in agent_self_correct(
                        agent=self,
                        original_task=f"Update file {file_id}",
                        current_context={},
                        error_details=str(e),
                        error_type="gdrive_update_error",
                        correction_guidance=f"Failed to update {file_desc} in GDrive. Ensure file ID {file_id} is correct.",
                    ):
                        pass  # Consume the iterator
                except Exception as self_correct_error:
                    logger.error(
                        f"[{self.name}] Self-correction also failed: {self_correct_error}"
                    )

            return False
    #
    # ========================================================================
    # Function 2.10: update_context
    # ========================================================================
    #
    def update_context(self, context: Dict) -> None:
        """Update the agent's current context with validation and logging."""
        # Validate context is a dictionary
        if not isinstance(context, dict):
            logger.warning(
                f"[{self.name}] Invalid context type: {type(context)}. Expected dict."
            )
            return

        try:
            # Store previous context size for logging
            previous_size = len(self.context) if self.context else 0

            # Update context
            self.context = (
                context.copy()
            )  # Create a copy to avoid external modifications

            # Log context update
            current_size = len(self.context)
            logger.debug(
                f"[{self.name}] Context updated: {previous_size} -> {current_size} keys"
            )

            # Log key changes if debug level
            if logger.level <= 10:  # DEBUG level
                context_keys = list(context.keys())
                logger.debug(f"[{self.name}] Context keys: {context_keys}")

        except Exception as e:
            logger.error(
                f"[{self.name}] Error updating context: {str(e)}", exc_info=True
            )
    #
    # ========================================================================
    # Function 2.11: run (Abstract Method)
    # ========================================================================
    #
    @abstractmethod
    async def run(
        self, task: str, context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        The main entry point for an agent to execute a task.

        This method must be implemented by all agent subclasses. It should:
        1. Process the input task and context
        2. Execute the agent's specific logic
        3. Yield streaming output chunks
        4. Handle errors gracefully
        5. Use the rule engine for output validation

        Args:
            task: The task description or instruction
            context: Optional context dictionary with additional parameters

        Yields:
            String chunks representing the agent's response

        Note:
            Implementations should use _execute_llm_workflow_with_rules() for
            LLM calls that need rule processing, or _execute_llm_workflow()
            for basic LLM calls without rules.
        """
        raise NotImplementedError("run() must be implemented by subclasses.")
#
# ============================================================================
# SECTION 3: Shared LLM Streaming Utility
# ============================================================================
# Function 3.1: _stream_llm_response
# ============================================================================
#
async def _stream_llm_response(
    model_name: str,
    contents: List[Any],
    generation_config: Optional[GenerationConfigType] = None,
) -> AsyncIterator[Any]:
    """Generic utility to stream responses from the Gemini API."""
    try:
        # Create model instance
        model = genai.GenerativeModel(model_name)

        # Generate content with streaming
        response: AsyncGenerateContentResponse = await model.generate_content_async(
            contents, stream=True, generation_config=generation_config
        )

        # Stream response chunks
        async for chunk in response:
            # Validate chunk has text content
            if not hasattr(chunk, "text") or not chunk.text:
                logger.warning(f"LLM stream returned a chunk with no text: {chunk}")
                continue
            yield chunk

    except StopCandidateException as e:
        error_message = f"[ERROR] LLM Stream Stopped: {str(e)}"
        logger.warning(error_message)
        yield error_message
    except Exception as e:
        error_message = f"[ERROR] LLM communication error: {str(e)}"
        logger.error(error_message, exc_info=True)
        yield error_message


#
#
## END: agent_base.py
