# ============================================================================
#  File: doc_agent.py
#  Version: 2.0 (Rule Engine Integrated)
#  Purpose: Documentation agent with integrated rule processing and automated
#           standards
#  Created: 30JUL25 | Updated: 04AUG25
# ============================================================================
# SECTION 1: Global Variable Definitions & Imports
# ============================================================================
#
import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from loguru import logger

from agents.agent_base import AgentBase
from src.error_handling import agent_self_correct
from src.telemetry import record_telemetry
#
# ============================================================================
# SECTION 2: fixtures
# ============================================================================
# Class 2.1: DocAgent
# Documentation agent with integrated rule processing and automated API
# documentation generation,
# ============================================================================
#
class DocAgent(AgentBase):
    #
    # ========================================================================
    # Method 2.1: __init__
    # ========================================================================
    #
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        websocket_manager=None,
        rule_engine=None,
        config_manager=None,
    ):
        super().__init__(name, config, websocket_manager, rule_engine, config_manager)

        # Documentation-specific configuration
        self.auto_generate_api_docs = config.get("auto_generate_api_docs", True)
        self.documentation_formats = config.get(
            "documentation_formats", ["markdown", "txt"]
        )
        self.include_examples = config.get("include_examples", True)
        self.compliance_standards = config.get(
            "compliance_standards", ["technical", "user-friendly"]
        )

        # Documentation quality thresholds
        self.min_section_count = config.get("min_section_count", 5)
        self.require_api_details = config.get("require_api_details", True)
        self.enforce_completeness = config.get("enforce_completeness", True)

    #
    # ========================================================================
    # Async Method 2.2: run
    # Execute documentation generation with integrated rule processing and
    # validation.
    # ========================================================================
    #
    @record_telemetry("DocAgent", "run")
    async def run(self, task: str, context: dict) -> AsyncIterator[str]:
        """
        Args:
            task: The documentation task description
            context: Execution context with file details, documentation type, etc.

        Yields:
            String chunks from the documentation generation process
        """
        # Update agent context
        self.update_context(context)

        # Log task initiation
        log_message = f"[{self.name}] Starting documentation task: {task}"
        logger.info(log_message)
        yield f"STREAM_CHUNK:{self.name}:{log_message}\n"

        try:
            # 1. Analyze documentation requirements and prepare context
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Analyzing documentation requirements...\n"
            doc_context = await self._prepare_documentation_context(task, context)

            # 2. Execute rule-integrated LLM workflow for documentation generation
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Generating documentation with rule validation...\n"
            full_response = ""

            # Use rule-integrated workflow for automatic validation and retries
            async for chunk in self._execute_llm_workflow_with_rules(
                prompt=doc_context["prompt"],
                task=task,
                context=context,
                max_rule_retries=3,
            ):
                # Check if this is an error chunk
                if "[ERROR]" in chunk:
                    yield chunk
                    return

                full_response += chunk
                yield f"STREAM_CHUNK:{self.name}:{chunk}"

            # 3. Post-generation processing and validation
            if full_response.strip():
                yield f"STREAM_CHUNK:{self.name}:[{self.name}] Processing documentation output...\n"
                await self._post_documentation_processing(
                    full_response, doc_context, task, context
                )

                # Final status
                yield f"STREAM_CHUNK:{self.name}:[{self.name}] Documentation generation completed successfully.\n"
            else:
                # Handle empty response
                error_message = f"[{self.name}] Generated documentation was empty"
                logger.error(error_message)
                yield f"STREAM_CHUNK:{self.name}:[ERROR] {error_message}\n"

        except Exception as e:
            # Use agent self-correction for error handling
            error_message = f"[{self.name}] Error in documentation generation: {str(e)}"
            logger.error(error_message, exc_info=True)

            # Attempt self-correction
            try:
                corrected_result = await agent_self_correct(
                    agent_name=self.name,
                    original_task=task,
                    error_context=str(e),
                    retry_count=1,
                )
                if corrected_result and corrected_result.get("success"):
                    yield f"STREAM_CHUNK:{self.name}:[{self.name}] Error corrected: {corrected_result['result']}\n"
                else:
                    yield f"STREAM_CHUNK:{self.name}:[ERROR] {error_message}\n"
            except Exception:
                yield f"STREAM_CHUNK:{self.name}:[ERROR] {error_message}\n"
    #
    # ========================================================================
    # Async Method 2.3: _prepare_documentation_context
    # ========================================================================
    #
    async def _prepare_documentation_context(
        self, task: str, context: Dict
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive context for documentation generation including code analysis.

        Args:
            task: The documentation generation task
            context: Input context

        Returns:
            Enhanced context dictionary for documentation prompt construction
        """
        # Base documentation context structure
        doc_context = {
            "task": task,
            "doc_type": context.get("doc_type", "comprehensive"),
            "format": context.get("format", "markdown"),
            "target_audience": context.get("target_audience", "developers"),
            "include_api": context.get("include_api", True),
            "include_examples": context.get("include_examples", True),
            "compliance_level": context.get("compliance_level", "standard"),
            "file_analysis": [],
        }

        # Read and analyze target files for documentation
        target_files = context.get("target_file_ids", [])
        if target_files:
            # Process each target file for comprehensive documentation coverage
            for i, file_id in enumerate(
                target_files[:5]
            ):  # Limit to prevent context overflow
                file_content = await self._read_gdrive_file(
                    file_id, f"source file {file_id}", task, context
                )

                if file_content:
                    # Analyze file for documentation requirements
                    file_analysis = await self._analyze_file_for_documentation(
                        file_content, file_id, task
                    )
                    doc_context["file_analysis"].append(file_analysis)

        # Select appropriate template based on context and rules
        template_name, template_content = await self.get_template_for_context(
            base_template_name="documentation_template", task=task, context=context
        )

        # Construct final prompt with all context
        doc_context["prompt"] = self._construct_prompt(
            template_name=template_name,
            task=task,
            doc_type=doc_context["doc_type"],
            format=doc_context["format"],
            target_audience=doc_context["target_audience"],
            file_analysis=doc_context["file_analysis"],
            compliance_standards=self.compliance_standards,
            quality_requirements=self._get_quality_requirements(doc_context),
        )

        return doc_context
    #
    # ========================================================================
    # Async Method 2.4: _analyze_file_for_documentation
    # ========================================================================
    #
    async def _analyze_file_for_documentation(
        self, file_content: str, file_id: str, task: str
    ) -> Dict[str, Any]:
        """
        Analyze a source file to extract documentation requirements and structure.

        Args:
            file_content: The content of the file to analyze
            file_id: The file identifier
            task: The documentation task

        Returns:
            Analysis results for documentation generation
        """
        analysis = {
            "file_id": file_id,
            "language": self._detect_language(file_content),
            "functions": [],
            "classes": [],
            "modules": [],
            "api_endpoints": [],
            "complexity_score": 0,
            "documentation_gaps": [],
        }

        try:
            # Language-specific analysis patterns
            if analysis["language"] == "python":
                # Extract Python functions, classes, and docstrings
                analysis.update(await self._analyze_python_file(file_content))
            elif analysis["language"] in ["javascript", "typescript"]:
                # Extract JS/TS functions, classes, and JSDoc
                analysis.update(await self._analyze_javascript_file(file_content))
            elif analysis["language"] == "powershell":
                # Extract PowerShell functions and help comments
                analysis.update(await self._analyze_powershell_file(file_content))
            else:
                # Generic analysis for other languages
                analysis.update(await self._analyze_generic_file(file_content))

            # Calculate documentation completeness score
            analysis["completeness_score"] = self._calculate_documentation_completeness(
                analysis
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error analyzing file {file_id}: {e}")
            analysis["error"] = str(e)

        return analysis
    #
    # ========================================================================
    # Async Method 2.5: _analyze_python_file
    # ========================================================================
    #
    async def _analyze_python_file(self, content: str) -> Dict[str, Any]:
        """
        Analyze Python source code for documentation requirements.

        Args:
            content: Python source code content

        Returns:
            Python-specific analysis results
        """
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "docstring_coverage": 0,
        }

        try:
            # Extract function definitions with signatures
            function_pattern = r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^:]+))?\s*:"
            functions = re.findall(function_pattern, content)

            for func_name, params, return_type in functions:
                # Check if function has docstring
                func_start = content.find(f"def {func_name}")
                if func_start != -1:
                    func_section = content[func_start : func_start + 500]
                    has_docstring = '"""' in func_section or "'''" in func_section

                    analysis["functions"].append(
                        {
                            "name": func_name,
                            "parameters": params.strip() if params else "",
                            "return_type": (
                                return_type.strip() if return_type else "None"
                            ),
                            "has_docstring": has_docstring,
                            "complexity": self._estimate_function_complexity(
                                func_section
                            ),
                        }
                    )

            # Extract class definitions
            class_pattern = r"class\s+(\w+)(?:\([^)]*\))?:"
            classes = re.findall(class_pattern, content)

            for class_name in classes:
                class_start = content.find(f"class {class_name}")
                if class_start != -1:
                    class_section = content[class_start : class_start + 300]
                    has_docstring = '"""' in class_section or "'''" in class_section

                    analysis["classes"].append(
                        {
                            "name": class_name,
                            "has_docstring": has_docstring,
                            "methods": self._extract_class_methods(content, class_name),
                        }
                    )

            # Calculate docstring coverage
            total_items = len(analysis["functions"]) + len(analysis["classes"])
            if total_items > 0:
                documented_items = sum(
                    1 for f in analysis["functions"] if f["has_docstring"]
                )
                documented_items += sum(
                    1 for c in analysis["classes"] if c["has_docstring"]
                )
                analysis["docstring_coverage"] = (documented_items / total_items) * 100

        except Exception as e:
            logger.error(f"[{self.name}] Error in Python file analysis: {e}")

        return analysis
    #
    # ========================================================================
    # Async Method 2.6: _analyze_javascript_file
    # ========================================================================
    #
    async def _analyze_javascript_file(self, content: str) -> Dict[str, Any]:
        """
        Analyze JavaScript/TypeScript source code for documentation requirements.

        Args:
            content: JavaScript/TypeScript source code content

        Returns:
            JavaScript-specific analysis results
        """
        analysis = {"functions": [], "classes": [], "exports": [], "jsdoc_coverage": 0}

        try:
            # Extract function declarations and expressions
            func_patterns = [
                r"function\s+(\w+)\s*\(([^)]*)\)",  # function declarations
                r"const\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>",  # arrow functions
                r"(\w+)\s*:\s*(?:async\s+)?function\s*\(([^)]*)\)",  # object methods
            ]

            for pattern in func_patterns:
                functions = re.findall(pattern, content)
                for func_name, params in functions:
                    # Check for JSDoc comment
                    func_start = content.find(func_name)
                    if func_start != -1:
                        # Look backwards for JSDoc comment
                        before_func = content[:func_start]
                        has_jsdoc = (
                            "/**" in before_func[-200:]
                            if len(before_func) >= 200
                            else "/**" in before_func
                        )

                        analysis["functions"].append(
                            {
                                "name": func_name,
                                "parameters": params.strip() if params else "",
                                "has_jsdoc": has_jsdoc,
                                "is_async": "async"
                                in content[max(0, func_start - 50) : func_start + 50],
                            }
                        )

            # Extract class definitions
            class_pattern = r"class\s+(\w+)(?:\s+extends\s+\w+)?"
            classes = re.findall(class_pattern, content)

            for class_name in classes:
                class_start = content.find(f"class {class_name}")
                if class_start != -1:
                    # Look for JSDoc before class
                    before_class = content[:class_start]
                    has_jsdoc = (
                        "/**" in before_class[-200:]
                        if len(before_class) >= 200
                        else "/**" in before_class
                    )

                    analysis["classes"].append(
                        {"name": class_name, "has_jsdoc": has_jsdoc}
                    )

            # Calculate JSDoc coverage
            total_items = len(analysis["functions"]) + len(analysis["classes"])
            if total_items > 0:
                documented_items = sum(
                    1 for f in analysis["functions"] if f["has_jsdoc"]
                )
                documented_items += sum(
                    1 for c in analysis["classes"] if c["has_jsdoc"]
                )
                analysis["jsdoc_coverage"] = (documented_items / total_items) * 100

        except Exception as e:
            logger.error(f"[{self.name}] Error in JavaScript file analysis: {e}")

        return analysis
    #
    # ========================================================================
    # Async Method 2.7: _analyze_powershell_file
    # ========================================================================
    #
    async def _analyze_powershell_file(self, content: str) -> Dict[str, Any]:
        """
        Analyze PowerShell source code for documentation requirements.

        Args:
            content: PowerShell source code content

        Returns:
            PowerShell-specific analysis results
        """
        analysis = {"functions": [], "cmdlets": [], "help_coverage": 0}

        try:
            # Extract function definitions
            func_pattern = r"function\s+(\w+(?:-\w+)*)\s*(?:\([^)]*\))?\s*{"
            functions = re.findall(func_pattern, content, re.IGNORECASE)

            for func_name in functions:
                func_start = content.lower().find(f"function {func_name.lower()}")
                if func_start != -1:
                    # Look for comment-based help
                    func_section = content[max(0, func_start - 500) : func_start + 300]
                    has_help = (
                        ".SYNOPSIS" in func_section or ".DESCRIPTION" in func_section
                    )

                    analysis["functions"].append(
                        {
                            "name": func_name,
                            "has_help": has_help,
                            "is_advanced": "[CmdletBinding()]" in func_section,
                        }
                    )

            # Calculate help coverage
            if analysis["functions"]:
                documented_funcs = sum(
                    1 for f in analysis["functions"] if f["has_help"]
                )
                analysis["help_coverage"] = (
                    documented_funcs / len(analysis["functions"])
                ) * 100

        except Exception as e:
            logger.error(f"[{self.name}] Error in PowerShell file analysis: {e}")

        return analysis
    #
    # ========================================================================
    # Async Method 2.8: _analyze_generic_file
    # ========================================================================
    #
    async def _analyze_generic_file(self, content: str) -> Dict[str, Any]:
        """
        Perform generic analysis for files in unrecognized languages.

        Args:
            content: Source code content

        Returns:
            Generic analysis results
        """
        analysis = {
            "line_count": len(content.splitlines()),
            "comment_ratio": 0,
            "has_headers": False,
        }

        try:
            lines = content.splitlines()
            comment_lines = 0

            # Count comment lines (basic patterns)
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("#", "//", "/*", "*", "--")):
                    comment_lines += 1

            if lines:
                analysis["comment_ratio"] = (comment_lines / len(lines)) * 100

            # Check for file headers
            first_10_lines = "\n".join(lines[:10])
            analysis["has_headers"] = any(
                indicator in first_10_lines.lower()
                for indicator in ["file:", "author:", "created:", "purpose:"]
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error in generic file analysis: {e}")

        return analysis
    #
    # ========================================================================
    # Method 2.9: _detect_language
    # ========================================================================
    #
    def _detect_language(self, content: str) -> str:
        """
        Detect the programming language of the file content.

        Args:
            content: File content to analyze

        Returns:
            Detected language identifier
        """
        # Language detection patterns
        if re.search(r"def\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import", content):
            return "python"
        elif re.search(
            r"function\s+\w+|const\s+\w+\s*=|let\s+\w+\s*=|\bclass\s+\w+", content
        ):
            return "javascript"
        elif re.search(r"interface\s+\w+|type\s+\w+\s*=|\bexport\s+", content):
            return "typescript"
        elif re.search(
            r"function\s+\w+(?:-\w+)*\s*{|\[CmdletBinding\(\)\]|Param\s*\(",
            content,
            re.IGNORECASE,
        ):
            return "powershell"
        elif re.search(r"#include\s*<|int\s+main\s*\(|void\s+\w+\s*\(", content):
            return "c"
        elif re.search(r"public\s+class\s+\w+|public\s+static\s+void\s+main", content):
            return "java"
        else:
            return "unknown"
    #
    # ========================================================================
    # Method 2.10: _get_quality_requirements
    # ========================================================================
    #
    def _get_quality_requirements(self, doc_context: Dict) -> Dict[str, Any]:
        """
        Determine quality requirements based on documentation context and agent configuration.

        Args:
            doc_context: Documentation context dictionary

        Returns:
            Quality requirements specification
        """
        return {
            "min_sections": self.min_section_count,
            "require_examples": self.include_examples,
            "api_documentation": self.require_api_details,
            "completeness_threshold": 80,
            "formats": self.documentation_formats,
            "compliance_standards": self.compliance_standards,
            "target_audience": doc_context.get("target_audience", "developers"),
        }
    #
    # ========================================================================
    # Async Method 2.11: _post_documentation_processing
    # ========================================================================
    #
    async def _post_documentation_processing(
        self, response: str, doc_context: Dict, task: str, context: Dict
    ) -> None:
        """
        Process generated documentation for quality assurance and file operations.

        Args:
            response: Generated documentation content
            doc_context: Documentation generation context
            task: Original task description
            context: Execution context
        """
        try:
            # Validate documentation completeness
            quality_score = self._assess_documentation_quality(response, doc_context)

            if quality_score < 70:
                logger.warning(
                    f"[{self.name}] Documentation quality score: {quality_score}% - below threshold"
                )
            else:
                logger.info(
                    f"[{self.name}] Documentation quality score: {quality_score}%"
                )

            # Save documentation if file operations are requested
            if context.get("save_to_file") and context.get("output_file_id"):
                await self._save_documentation_file(
                    response, context["output_file_id"], doc_context["format"]
                )

        except Exception as e:
            logger.error(f"[{self.name}] Error in post-documentation processing: {e}")
    #
    # ========================================================================
    # Method 2.12: _assess_documentation_quality
    # ========================================================================
    #
    def _assess_documentation_quality(
        self, documentation: str, doc_context: Dict
    ) -> int:
        """
        Assess the quality of generated documentation against requirements.

        Args:
            documentation: Generated documentation content
            doc_context: Documentation context with requirements

        Returns:
            Quality score (0-100)
        """
        score = 0
        total_checks = 0

        try:
            # Check for required sections
            required_sections = ["overview", "installation", "usage", "api", "examples"]
            sections_found = 0

            for section in required_sections:
                if section.lower() in documentation.lower():
                    sections_found += 1

            if required_sections:
                score += (sections_found / len(required_sections)) * 30
                total_checks += 30

            # Check documentation length (should be substantial)
            if len(documentation) > 500:
                score += 20
            elif len(documentation) > 200:
                score += 10
            total_checks += 20

            # Check for code examples
            if "```" in documentation or "    " in documentation:
                score += 15
            total_checks += 15

            # Check for API documentation elements
            api_indicators = ["parameters", "returns", "example", "usage"]
            api_score = sum(
                5 for indicator in api_indicators if indicator in documentation.lower()
            )
            score += min(api_score, 20)
            total_checks += 20

            # Check formatting quality
            if "##" in documentation or "**" in documentation:
                score += 10
            total_checks += 10

            # Normalize score
            if total_checks > 0:
                score = min(100, (score / total_checks) * 100)

        except Exception as e:
            logger.error(f"[{self.name}] Error assessing documentation quality: {e}")
            score = 50  # Default middle score on error

        return int(score)
    #
    # ========================================================================
    # Async Method 2.13: _save_documentation_file
    # ========================================================================
    #
    async def _save_documentation_file(
        self, content: str, file_id: str, format_type: str
    ) -> None:
        """
        Save generated documentation to the specified file.

        Args:
            content: Documentation content to save
            file_id: Target file ID for saving
            format_type: Documentation format (markdown, txt, etc.)
        """
        try:
            # Format content based on type
            if format_type.lower() == "markdown":
                # Ensure proper markdown formatting
                if not content.startswith("#"):
                    content = "# Documentation\n\n" + content

            # Save using inherited GDrive functionality
            await self._save_gdrive_file(
                file_id, content, f"Documentation ({format_type})"
            )
            logger.info(f"[{self.name}] Documentation saved to file: {file_id}")

        except Exception as e:
            logger.error(f"[{self.name}] Error saving documentation file: {e}")
    #
    # ========================================================================
    # Method 2.14: _estimate_function_complexity
    # ========================================================================
    #
    def _estimate_function_complexity(self, func_content: str) -> str:
        """
        Estimate the complexity level of a function for documentation prioritization.

        Args:
            func_content: Function source code content

        Returns:
            Complexity level (simple, moderate, complex)
        """
        try:
            # Count complexity indicators
            complexity_indicators = [
                len(re.findall(r"\bif\b", func_content)),  # Conditional statements
                len(re.findall(r"\bfor\b", func_content)),  # Loops
                len(re.findall(r"\bwhile\b", func_content)),  # While loops
                len(re.findall(r"\btry\b", func_content)),  # Exception handling
                len(re.findall(r"\bwith\b", func_content)),  # Context managers
            ]

            total_complexity = sum(complexity_indicators)

            if total_complexity >= 5:
                return "complex"
            elif total_complexity >= 2:
                return "moderate"
            else:
                return "simple"

        except Exception:
            return "moderate"  # Default to moderate on error

    #
    # ========================================================================
    #    Method 2.15: _extract_class_methods
    # ========================================================================
    #
    def _extract_class_methods(self, content: str, class_name: str) -> List[Dict]:
        """
        Extract method information from a class definition.

        Args:
            content: Source code content
            class_name: Name of the class to analyze

        Returns:
            List of method information dictionaries
        """
        methods = []

        try:
            # Find class definition start
            class_start = content.find(f"class {class_name}")
            if class_start == -1:
                return methods

            # Find next class or end of file to limit search scope
            remaining_content = content[class_start:]
            next_class = re.search(
                r"\nclass\s+\w+", remaining_content[50:]
            )  # Skip current class

            if next_class:
                class_content = remaining_content[: next_class.start() + 50]
            else:
                class_content = remaining_content

            # Extract method definitions within the class
            method_pattern = r"def\s+(\w+)\s*\(([^)]*)\)"
            method_matches = re.findall(method_pattern, class_content)

            for method_name, params in method_matches:
                # Skip if it's not actually a method (could be nested function)
                method_start = class_content.find(f"def {method_name}")
                if method_start != -1:
                    # Check indentation to confirm it's a class method
                    lines_before = class_content[:method_start].split("\n")
                    if lines_before:
                        # Simple heuristic: method should be indented within class
                        method_line_start = method_start - len(lines_before[-1])
                        if method_line_start > 0:  # Has some indentation
                            methods.append(
                                {
                                    "name": method_name,
                                    "parameters": params.strip() if params else "",
                                    "is_private": method_name.startswith("_"),
                                    "is_special": method_name.startswith("__")
                                    and method_name.endswith("__"),
                                }
                            )

        except Exception as e:
            logger.error(f"[{self.name}] Error extracting class methods: {e}")

        return methods
    #
    # ========================================================================
    # Method 2.16: _calculate_documentation_completeness
    # ========================================================================
    #
    def _calculate_documentation_completeness(self, analysis: Dict) -> int:
        """
        Calculate overall documentation completeness score for a file.

        Args:
            analysis: File analysis results

        Returns:
            Completeness score (0-100)
        """
        try:
            score = 0

            # Base score for having any documentation
            if (
                analysis.get("docstring_coverage", 0) > 0
                or analysis.get("jsdoc_coverage", 0) > 0
                or analysis.get("help_coverage", 0) > 0
            ):
                score += 20

            # Score based on documentation coverage percentage
            coverage = max(
                analysis.get("docstring_coverage", 0),
                analysis.get("jsdoc_coverage", 0),
                analysis.get("help_coverage", 0),
            )
            score += min(60, coverage * 0.6)  # Up to 60 points for coverage

            # Bonus for good comment ratio in generic files
            comment_ratio = analysis.get("comment_ratio", 0)
            if comment_ratio > 10:
                score += min(10, comment_ratio * 0.5)

            # Bonus for file headers
            if analysis.get("has_headers", False):
                score += 10

            return min(100, int(score))

        except Exception:
            return 50  # Default score on calculation error
#
#
## End of doc_agent.py
