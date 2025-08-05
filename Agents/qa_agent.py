# ============================================================================
#  File: qa_agent.py
#  Version: 2.0 (Rule Engine Integrated)
#  Purpose: Quality Assurance agent with security scanning and compliance
#  Created: 30JUL25 | Updated: 04AUG25
# ============================================================================
# SECTION 1: Global Variable Definitions & Imports
# ============================================================================
#
import asyncio
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Set
from loguru import logger

from agents.agent_base import AgentBase
from src.error_handling import agent_self_correct
from src.telemetry import record_telemetry
#
# ============================================================================
# SECTION 2: Fixtures
# ============================================================================
# Class 2.1: QAAgent Class
# Quality Assurance agent with integrated rule processing, security scanning,
# compliance checking, and automated review workflows.
# ============================================================================
#
class QAAgent(AgentBase):
    #
    # ============================================================================
    # Method 2.1.1: __init__
    # ============================================================================
    #
    def __init__(self, name: str, config: Dict[str, Any], websocket_manager=None,
                 rule_engine=None, config_manager=None):
        super().__init__(name, config, websocket_manager, rule_engine, config_manager)

        # QA-specific configuration
        self.security_scanning = config.get("security_scanning", True)
        self.compliance_checking = config.get("compliance_checking", True)
        self.code_quality_analysis = config.get("code_quality_analysis", True)
        self.performance_analysis = config.get("performance_analysis", True)
        self.documentation_review = config.get("documentation_review", True)
    #
    # ============================================================================
    # Method 2.1.2: run
    # ============================================================================
    #
    @record_telemetry("QAAgent", "run")
    async def run(self, task: str, context: dict) -> AsyncIterator[str]:
        """
        Execute quality assurance with integrated rule processing and validation.

        Args:
            task: The QA task description
            context: Execution context with files to review, standards, etc.

        Yields:
            String chunks from the QA process
        """
        # Update agent context
        self.update_context(context)

        # Log task initiation
        log_message = f"[{self.name}] Starting QA review task: {task}"
        logger.info(log_message)
        yield f"STREAM_CHUNK:{self.name}:{log_message}\n"

        # QA process execution - comprehensive error handling required
        try:
            # 1. Prepare QA context and analyze target files
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Preparing QA analysis context...\n"
            qa_context = await self._prepare_qa_context(task, context)

            # 2. Get context-aware template (may include rule-based overrides)
            base_template = self.config.get("qa_template", "base_qa_prompt.txt")
            template_name, template_content = await self.get_template_for_context(
                base_template, task, qa_context
            )

            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Using QA template: {template_name}\n"
            logger.info(f"[{self.name}] Selected template: {template_name}")

            # 3. Construct QA prompt with comprehensive context
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Constructing QA review prompt...\n"
            prompt = self._construct_prompt(template_name, **qa_context)

            # 4. Execute QA review with rule-integrated workflow
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Executing QA review with rule validation...\n"

            # Use rule-integrated LLM workflow for automatic validation and retry
            async for chunk in self._execute_llm_workflow_with_rules(
                prompt=prompt,
                task=task,
                context=qa_context,
                max_rule_retries=2
            ):
                yield chunk

            # 5. Post-QA operations and reporting
            await self._handle_post_qa_operations(task, qa_context)

            # Success notification
            success_msg = f"[SUCCESS] [{self.name}] QA review completed successfully."
            logger.info(success_msg)
            yield f"STREAM_CHUNK:{self.name}:{success_msg}\n"

        except Exception as e:
            # Enhanced error handling with self-correction
            error_message = f"[{self.name}] QA review failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            yield f"STREAM_CHUNK:{self.name}:{error_message}\n"

            # Trigger self-correction workflow for QA failures
            async for chunk in agent_self_correct(
                agent=self,
                original_task=task,
                current_context=context,
                error_details=str(e),
                error_type="qa_review_error",
                correction_guidance="QA review failed. Analyze the requirements and provide alternative review approach."
            ):
                yield chunk
    #
    # ============================================================================
    # Async Method 2.1.3: _prepare_qa_context
    # ============================================================================
    #
    async def _prepare_qa_context(self, task: str, context: Dict) -> Dict[str, Any]:
        """
        Prepare comprehensive context for QA review including file analysis.

        Args:
            task: The QA task
            context: Input context

        Returns:
            Enhanced context dictionary for QA prompt construction
        """
        # Base QA context structure
        qa_context = {
            "task": task,
            "review_type": context.get("review_type", "comprehensive"),
            "language": context.get("language", "python"),
            "project_type": context.get("project_type", "general"),
            "compliance_standards": context.get("compliance_standards", []),
            "security_requirements": context.get("security_requirements", []),
            "performance_targets": context.get("performance_targets", {}),
            "code_standards": context.get("code_standards", "standard")
        }

        # Read target files for review - handle multiple file analysis
        target_files = context.get("target_file_ids", [])
        if target_files:
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Reading {len(target_files)} files for QA review...\n"
            file_contents = []

            # Process each target file for comprehensive analysis
            for i, file_id in enumerate(target_files[:5]):  # Limit to prevent context overflow
                yield f"STREAM_CHUNK:{self.name}:[{self.name}] Analyzing file {i+1}/{min(len(target_files), 5)}...\n"
                file_content = await self._read_gdrive_file(
                    file_id, f"target file {file_id}", task, context
                )

                # Store file content with metadata for analysis
                if file_content:
                    file_analysis = await self._analyze_file_for_qa(file_content, file_id, qa_context)
                    file_contents.append(file_analysis)

            qa_context["target_files"] = file_contents

        # Enhanced context based on review type
        review_type = qa_context["review_type"]

        # Security-focused QA review context
        if review_type in ["security", "comprehensive"]:
            qa_context["security_checklist"] = await self._get_security_checklist(qa_context["language"])
            qa_context["vulnerability_patterns"] = await self._get_vulnerability_patterns(qa_context["language"])

        # Performance-focused QA review context
        if review_type in ["performance", "comprehensive"]:
            qa_context["performance_checklist"] = await self._get_performance_checklist(qa_context["language"])
            qa_context["optimization_patterns"] = await self._get_optimization_patterns(qa_context["language"])

        # Code quality-focused QA review context
        if review_type in ["quality", "comprehensive"]:
            qa_context["quality_standards"] = await self._get_quality_standards(qa_context["language"])
            qa_context["best_practices"] = await self._get_best_practices(qa_context["language"])

        # Language-specific QA enhancements
        language = qa_context["language"].lower()

        # Python-specific QA context
        if language == "python":
            qa_context.update({
                "python_version": context.get("python_version", "3.13+"),
                "style_guide": context.get("style_guide", "PEP 8"),
                "testing_framework": context.get("testing_framework", "pytest"),
                "linting_tools": ["pylint", "flake8", "mypy", "bandit"],
                "security_tools": ["bandit", "safety", "semgrep"]
            })

        # JavaScript/TypeScript-specific QA context
        elif language in ["javascript", "typescript"]:
            qa_context.update({
                "js_standard": context.get("js_standard", "ES6+"),
                "style_guide": context.get("style_guide", "ESLint"),
                "testing_framework": context.get("testing_framework", "Jest"),
                "linting_tools": ["eslint", "prettier", "tslint"],
                "security_tools": ["eslint-plugin-security", "audit"]
            })

        # PowerShell-specific QA context
        elif language == "powershell":
            qa_context.update({
                "ps_version": context.get("ps_version", "5.1+"),
                "execution_policy": context.get("execution_policy", "RemoteSigned"),
                "testing_framework": context.get("testing_framework", "Pester"),
                "linting_tools": ["PSScriptAnalyzer", "PSCodeHealth"],
                "security_tools": ["PSScriptAnalyzer"]
            })

        logger.debug(f"[{self.name}] QA context prepared with {len(qa_context)} parameters")
        return qa_context
    #
    # ============================================================================
    # Async Method 2.1.4: _analyze_file_for_qa
    # ============================================================================
    #
    async def _analyze_file_for_qa(self, file_content: str, file_id: str, context: Dict) -> Dict[str, Any]:
        """
        Analyze individual file for QA review.

        Args:
            file_content: Content of the file
            file_id: File identifier
            context: QA context

        Returns:
            File analysis results
        """
        analysis = {
            "file_id": file_id,
            "content": file_content[:2000],  # Truncate for context management
            "line_count": len(file_content.split('\n')),
            "char_count": len(file_content),
            "language": context.get("language", "unknown"),
            "issues_found": [],
            "security_flags": [],
            "performance_notes": [],
            "quality_scores": {}
        }

        # Basic code analysis patterns
        language = context.get("language", "").lower()

        # Security pattern detection based on language
        if language == "python":
            # Check for common Python security issues
            security_patterns = [
                (r'eval\s*\(', "Use of eval() function"),
                (r'exec\s*\(', "Use of exec() function"),
                (r'os\.system\s*\(', "Use of os.system()"),
                (r'subprocess\.call\s*\(.*shell\s*=\s*True', "Shell injection risk"),
                (r'pickle\.loads?\s*\(', "Unsafe pickle usage"),
                (r'yaml\.load\s*\(', "Unsafe YAML loading")
            ]

            # Check each security pattern
            for pattern, description in security_patterns:
                if re.search(pattern, file_content, re.IGNORECASE):
                    analysis["security_flags"].append(description)

        elif language in ["javascript", "typescript"]:
            # Check for JavaScript/TypeScript security issues
            security_patterns = [
                (r'eval\s*\(', "Use of eval() function"),
                (r'innerHTML\s*=', "innerHTML assignment (XSS risk)"),
                (r'document\.write\s*\(', "Use of document.write()"),
                (r'Function\s*\(', "Dynamic function creation"),
                (r'setTimeout\s*\(\s*["\']', "String-based setTimeout"),
                (r'setInterval\s*\(\s*["\']', "String-based setInterval")
            ]

            # Check each security pattern
            for pattern, description in security_patterns:
                if re.search(pattern, file_content, re.IGNORECASE):
                    analysis["security_flags"].append(description)

        # Performance pattern detection
        performance_patterns = {
            "python": [
                (r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(', "Inefficient range(len()) usage"),
                (r'\.append\s*\(.*\)\s*for\s+.*\s+in', "List comprehension opportunity"),
                (r'open\s*\([^)]*\)\s*(?!.*with)', "File not using context manager")
            ],
            "javascript": [
                (r'for\s*\(\s*var\s+\w+\s*=.*\.length', "Inefficient for loop"),
                (r'innerHTML\s*\+=', "Inefficient DOM manipulation"),
                (r'document\.getElementById.*loop', "Repeated DOM queries")
            ]
        }

        # Check performance patterns if available for language
        if language in performance_patterns:
            for pattern, description in performance_patterns[language]:
                if re.search(pattern, file_content, re.IGNORECASE):
                    analysis["performance_notes"].append(description)

        # Quality scoring based on various metrics
        quality_metrics = {
            "has_docstrings": bool(re.search(r'""".*?"""', file_content, re.DOTALL)),
            "has_comments": bool(re.search(r'#.*\S', file_content)),
            "reasonable_line_length": len([line for line in file_content.split('\n') if len(line) > 120]) < 5,
            "has_error_handling": bool(re.search(r'try\s*:|except\s*:', file_content)),
            "has_type_hints": bool(re.search(r':\s*\w+\s*=|def.*->', file_content))
        }

        analysis["quality_scores"] = quality_metrics

        return analysis
    #
    # ============================================================================
    # Async Method 2.1.5: _get_security_checklist
    # ============================================================================
    #
    async def _get_security_checklist(self, language: str) -> List[str]:
        """Get language-specific security checklist."""
        checklists = {
            "python": [
                "Input validation and sanitization",
                "SQL injection prevention",
                "Path traversal protection",
                "Pickle/eval usage review",
                "Authentication and authorization",
                "Secure error handling",
                "Dependency vulnerability check",
                "Secrets management review"
            ],
            "javascript": [
                "XSS prevention measures",
                "CSRF protection implementation",
                "Input validation and encoding",
                "Safe DOM manipulation",
                "Authentication token security",
                "Third-party library security",
                "Content Security Policy",
                "Secure communication protocols"
            ],
            "powershell": [
                "Execution policy compliance",
                "Script injection prevention",
                "Privilege escalation checks",
                "Credential handling security",
                "Remote execution security",
                "Input validation patterns",
                "Logging and auditing",
                "Module import security"
            ]
        }

        return checklists.get(language.lower(), [
            "General input validation",
            "Authentication checks",
            "Authorization verification",
            "Error handling security",
            "Data encryption review"
        ])
    #
    # ============================================================================
    # Async Method 2.1.6: _get_vulnerability_patterns
    # ============================================================================
    #
    async def _get_vulnerability_patterns(self, language: str) -> Dict[str, str]:
        """Get common vulnerability patterns for language."""
        patterns = {
            "python": {
                "sql_injection": r"execute\s*\(\s*[\"'].*%.*[\"']\s*%",
                "command_injection": r"os\.system\s*\(\s*.*\+.*\)",
                "path_traversal": r"open\s*\(\s*.*\+.*\)",
                "unsafe_deserialization": r"pickle\.loads?\s*\(",
                "code_injection": r"eval\s*\(\s*.*input"
            },
            "javascript": {
                "xss_vulnerability": r"innerHTML\s*=\s*.*\+",
                "code_injection": r"eval\s*\(\s*.*input",
                "prototype_pollution": r"__proto__\s*\[",
                "unsafe_redirect": r"location\s*=\s*.*input",
                "dom_clobbering": r"document\.\w+\s*=\s*.*input"
            },
            "powershell": {
                "command_injection": r"Invoke-Expression\s*.*\$",
                "script_injection": r"iex\s*.*\$",
                "privilege_escalation": r"Start-Process.*-Verb\s+RunAs",
                "unsafe_execution": r"&\s*\$.*input",
                "credential_exposure": r"ConvertTo-SecureString.*PlainText"
            }
        }

        return patterns.get(language.lower(), {})
    #
    # ============================================================================
    # Async Method 2.1.7: get_performance_checklist
    # ============================================================================
    #
    async def _get_performance_checklist(self, language: str) -> List[str]:
        """Get performance optimization checklist."""
        checklists = {
            "python": [
                "List comprehensions vs loops",
                "Generator usage for large datasets",
                "Efficient string operations",
                "Proper data structure selection",
                "Caching and memoization",
                "Database query optimization",
                "Memory usage patterns",
                "Algorithmic complexity review"
            ],
            "javascript": [
                "DOM manipulation efficiency",
                "Event delegation patterns",
                "Async/await vs callbacks",
                "Memory leak prevention",
                "Bundle size optimization",
                "Lazy loading implementation",
                "Caching strategies",
                "Algorithmic efficiency"
            ],
            "powershell": [
                "Pipeline optimization",
                "WMI vs CIM cmdlets",
                "ForEach-Object vs foreach",
                "Array vs ArrayList usage",
                "Remote execution efficiency",
                "Memory consumption patterns",
                "Filter vs Where-Object",
                "Module loading optimization"
            ]
        }

        return checklists.get(language.lower(), [
            "Algorithm efficiency review",
            "Memory usage optimization",
            "I/O operation efficiency",
            "Resource cleanup patterns",
            "Caching implementation"
        ])
    #
    # ============================================================================
    # Async Method 2.1.8: _get_optimization_patterns
    # ============================================================================
    #
    async def _get_optimization_patterns(self, language: str) -> Dict[str, str]:
        """Get optimization opportunity patterns."""
        patterns = {
            "python": {
                "list_comprehension": r"for\s+\w+\s+in\s+.*:\s*\w+\.append",
                "generator_opportunity": r"return\s*\[.*for.*in.*\]",
                "string_concatenation": r"\+=\s*[\"'].*[\"']",
                "inefficient_loop": r"for\s+\w+\s+in\s+range\s*\(\s*len",
                "repeated_calculation": r"for.*:\s*.*\.calculate\("
            },
            "javascript": {
                "dom_query_optimization": r"document\.getElementById.*for",
                "array_method_chaining": r"\.map\(.*\)\.filter\(.*\)",
                "inefficient_loop": r"for\s*\(\s*var\s+\w+\s*=.*\.length",
                "repeated_dom_access": r"\.innerHTML\s*\+=",
                "callback_optimization": r"function.*callback.*for"
            }
        }

        return patterns.get(language.lower(), {})
    #
    # ============================================================================
    # Async Method 2.1.9: _get_quality_standards
    # ============================================================================
    #
    async def _get_quality_standards(self, language: str) -> Dict[str, Any]:
        """Get code quality standards for language."""
        standards = {
            "python": {
                "style_guide": "PEP 8",
                "docstring_format": "Google or NumPy style",
                "type_hints": "Required for public APIs",
                "line_length": 88,
                "complexity_limit": 10,
                "test_coverage": 80
            },
            "javascript": {
                "style_guide": "ESLint recommended",
                "documentation": "JSDoc format",
                "type_checking": "TypeScript preferred",
                "line_length": 100,
                "complexity_limit": 10,
                "test_coverage": 80
            },
            "powershell": {
                "style_guide": "PowerShell best practices",
                "documentation": "Comment-based help",
                "error_handling": "Try-catch blocks",
                "line_length": 115,
                "complexity_limit": 15,
                "test_coverage": 70
            }
        }

        return standards.get(language.lower(), {
            "style_guide": "Language best practices",
            "documentation": "Comprehensive comments",
            "error_handling": "Proper exception handling",
            "line_length": 100,
            "complexity_limit": 10,
            "test_coverage": 75
        })
    #
    # ============================================================================
    # Async Method 2.1.10: _get_best_practices
    # ============================================================================
    #
    async def _get_best_practices(self, language: str) -> List[str]:
        """Get best practices for language."""
        practices = {
            "python": [
                "Use virtual environments",
                "Follow PEP 8 style guide",
                "Write comprehensive docstrings",
                "Use type hints for clarity",
                "Implement proper error handling",
                "Use context managers for resources",
                "Prefer list comprehensions",
                "Use meaningful variable names"
            ],
            "javascript": [
                "Use strict mode",
                "Avoid global variables",
                "Use const/let over var",
                "Implement error boundaries",
                "Use async/await properly",
                "Optimize DOM operations",
                "Use proper event handling",
                "Implement proper testing"
            ],
            "powershell": [
                "Use approved verbs",
                "Implement proper error handling",
                "Use parameter validation",
                "Write comment-based help",
                "Use pipeline efficiently",
                "Avoid aliases in scripts",
                "Use proper scoping",
                "Implement comprehensive logging"
            ]
        }

        return practices.get(language.lower(), [
            "Follow language conventions",
            "Implement proper error handling",
            "Write clear documentation",
            "Use meaningful names",
            "Optimize for readability",
            "Test thoroughly",
            "Handle edge cases",
            "Follow security practices"
        ])
    #
    # ============================================================================
    # Async Method 2.1.11: _handle_post_qa_operations
    # ============================================================================
    #
    async def _handle_post_qa_operations(self, task: str, context: Dict) -> None:
        """
        Handle post-QA operations like report generation and metrics.

        Args:
            task: The original task
            context: QA context
        """
        # Post-QA processing required for comprehensive reporting
        try:
            # Log QA completion statistics
            review_type = context.get("review_type", "comprehensive")
            target_files = context.get("target_files", [])
            language = context.get("language", "unknown")

            logger.info(f"[{self.name}] QA review completed - Type: {review_type}, Files: {len(target_files)}, Language: {language}")

            # Generate QA metrics summary
            total_issues = sum(len(file_info.get("issues_found", [])) for file_info in target_files)
            total_security_flags = sum(len(file_info.get("security_flags", [])) for file_info in target_files)
            total_performance_notes = sum(len(file_info.get("performance_notes", [])) for file_info in target_files)

            qa_summary = {
                "total_files_reviewed": len(target_files),
                "total_issues_found": total_issues,
                "security_flags": total_security_flags,
                "performance_notes": total_performance_notes,
                "review_type": review_type,
                "language": language
            }

            logger.info(f"[{self.name}] QA Summary: {qa_summary}")

        except Exception as e:
            logger.warning(f"[{self.name}] Post-QA operations had issues: {str(e)}")
    #
    # ============================================================================
    # Async Method 2.1.12: generate_qa_report
    # ============================================================================
    #
    async def generate_qa_report(self, task: str, context: Dict) -> Optional[str]:
        """
        Generate comprehensive QA report and save to Google Drive.

        Args:
            task: QA review task
            context: Must include target files and optionally output settings

        Returns:
            File ID if successful, None otherwise
        """
        # Comprehensive QA report generation with error handling
        try:
            # Validate required context for report generation
            target_files = context.get("target_file_ids", [])
            if not target_files:
                logger.error(f"[{self.name}] Missing target_file_ids for QA report generation")
                return None

            # Buffer the QA report content
            qa_report = ""
            async for chunk in self.run(task, context):
                if not chunk.startswith("STREAM_CHUNK:"):
                    qa_report += chunk

            # Save QA report if content generated
            if qa_report.strip():
                output_filename = context.get("output_filename", f"qa_report_{task[:20]}.md")
                parent_folder_id = context.get("parent_folder_id")

                if parent_folder_id:
                    file_id = await self._write_gdrive_file(output_filename, qa_report, parent_folder_id)
                    if file_id:
                        logger.info(f"[{self.name}] QA report saved with ID: {file_id}")
                        return file_id
                    else:
                        logger.error(f"[{self.name}] Failed to save QA report")
                        return None
                else:
                    logger.warning(f"[{self.name}] No parent folder specified for QA report")
                    return None
            else:
                logger.warning(f"[{self.name}] No QA report content generated")
                return None

        except Exception as e:
            logger.error(f"[{self.name}] Error in generate_qa_report: {str(e)}", exc_info=True)
            return None
    #
    # ============================================================================
    # Async Method 2.1.13: get_supported_review_types
    # ============================================================================
    #
    async def get_supported_review_types(self) -> List[str]:
        """
        Get list of supported QA review types.

        Returns:
            List of review type identifiers
        """
        return [
            "comprehensive",
            "security",
            "performance",
            "quality",
            "compliance",
            "documentation",
            "accessibility",
            "maintainability"
        ]
    #
    # ============================================================================
    # Async Method 2.1.14: get_compliance_standards
    # ============================================================================
    #
    async def get_compliance_standards(self) -> Dict[str, List[str]]:
        """
        Get available compliance standards by domain.

        Returns:
            Dictionary of compliance standards by domain
        """
        return {
            "security": [
                "OWASP Top 10",
                "CWE/SANS Top 25",
                "NIST Cybersecurity Framework",
                "ISO 27001",
                "SOC 2"
            ],
            "privacy": [
                "GDPR",
                "CCPA",
                "HIPAA",
                "PCI DSS"
            ],
            "code_quality": [
                "ISO/IEC 25010",
                "IEEE Standards",
                "Clean Code Principles",
                "SOLID Principles"
            ],
            "accessibility": [
                "WCAG 2.1",
                "Section 508",
                "ADA Compliance"
            ]
        }
#
#
## END: qa_agent.py
