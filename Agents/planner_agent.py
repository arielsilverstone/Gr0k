# ============================================================================
#  File: planner_agent.py
#  Version: 2.0 (Rule Engine Integrated)
#  Purpose: Planning agent with integrated rule processing and project estimation
#  Created: 30JUL25 | Updated: 04AUG25
# ============================================================================
# SECTION 1: Global Variable Definitions & Imports
# ============================================================================
#
import asyncio
import re
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from loguru import logger

from agents.agent_base import AgentBase
from src.error_handling import agent_self_correct
from src.telemetry import record_telemetry
#
# ============================================================================
# SECTION 2: PlannerAgent Class
# ============================================================================
# Class 2.1: PlannerAgent
# Planning agent with integrated rule processing, project estimation
# capabilities, resource allocation planning, timeline management, and
# dependency analysis.
# ============================================================================
#
class PlannerAgent(AgentBase):
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

        # Planning-specific configuration
        self.default_estimation_method = config.get("estimation_method", "story_points")
        self.complexity_factors = config.get(
            "complexity_factors", ["technical", "integration", "testing"]
        )
        self.resource_types = config.get(
            "resource_types", ["developer", "qa", "devops", "design"]
        )
        self.risk_assessment_enabled = config.get("risk_assessment", True)

        # Planning quality thresholds
        self.min_plan_sections = config.get("min_plan_sections", 6)
        self.require_timeline = config.get("require_timeline", True)
        self.require_dependencies = config.get("require_dependencies", True)
        self.require_risk_analysis = config.get("require_risk_analysis", True)
    #
    # ========================================================================
    # Method 2.2: run
    # ========================================================================
    #
    @record_telemetry("PlannerAgent", "run")
    async def run(self, task: str, context: dict) -> AsyncIterator[str]:
        """
        Execute project planning with integrated rule processing and validation.

        Args:
            task: The planning task description
            context: Execution context with project details, scope, requirements, etc.

        Yields:
            String chunks from the planning process
        """
        # Update agent context
        self.update_context(context)

        # Log task initiation
        log_message = f"[{self.name}] Starting planning task: {task}"
        logger.info(log_message)
        yield f"STREAM_CHUNK:{self.name}:{log_message}\n"

        try:
            # 1. Analyze planning requirements and prepare comprehensive context
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Analyzing project requirements and scope...\n"
            planning_context = await self._prepare_planning_context(task, context)

            # 2. Execute rule-integrated LLM workflow for plan generation
            yield f"STREAM_CHUNK:{self.name}:[{self.name}] Generating comprehensive project plan with rule validation...\n"
            full_response = ""

            # Use rule-integrated workflow for automatic validation and retries
            async for chunk in self._execute_llm_workflow_with_rules(
                prompt=planning_context["prompt"],
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

            # 3. Post-planning processing and validation
            if full_response.strip():
                yield f"STREAM_CHUNK:{self.name}:[{self.name}] Processing plan output and generating estimates...\n"
                await self._post_planning_processing(
                    full_response, planning_context, task, context
                )

                # Final status
                yield f"STREAM_CHUNK:{self.name}:[{self.name}] Project planning completed successfully.\n"
            else:
                # Handle empty response
                error_message = f"[{self.name}] Generated plan was empty"
                logger.error(error_message)
                yield f"STREAM_CHUNK:{self.name}:[ERROR] {error_message}\n"

        except Exception as e:
            # Use agent self-correction for error handling
            error_message = f"[{self.name}] Error in project planning: {str(e)}"
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
    # Method 2.3: prepare_planning_context
    # ========================================================================
    #
    async def _prepare_planning_context(
        self, task: str, context: Dict
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive context for project planning including requirements analysis.

        Args:
            task: The planning task description
            context: Input context

        Returns:
            Enhanced context dictionary for planning prompt construction
        """
        # Base planning context structure
        planning_context = {
            "task": task,
            "planning_type": context.get("planning_type", "comprehensive"),
            "scope": context.get("scope", "feature"),
            "complexity_level": context.get("complexity_level", "moderate"),
            "timeline_constraint": context.get("timeline_constraint", "flexible"),
            "team_size": context.get("team_size", "small"),
            "project_phase": context.get("project_phase", "development"),
            "requirements_analysis": [],
            "dependencies_analysis": [],
            "risk_factors": [],
        }

        # Analyze project requirements from input files or context
        target_files = context.get("target_file_ids", [])
        if target_files:
            # Process requirement documents or specifications
            for i, file_id in enumerate(
                target_files[:3]
            ):  # Limit to prevent context overflow
                file_content = await self._read_gdrive_file(
                    file_id, f"requirement file {file_id}", task, context
                )

                if file_content:
                    # Analyze requirements for planning insights
                    req_analysis = await self._analyze_requirements_document(
                        file_content, file_id, task
                    )
                    planning_context["requirements_analysis"].append(req_analysis)

        # Perform dependency analysis
        planning_context["dependencies_analysis"] = (
            await self._analyze_project_dependencies(planning_context, task)
        )

        # Assess project risks
        if self.risk_assessment_enabled:
            planning_context["risk_factors"] = await self._assess_project_risks(
                planning_context, task
            )

        # Generate effort estimates
        planning_context["effort_estimates"] = await self._generate_effort_estimates(
            planning_context, task
        )

        # Select appropriate template based on context and rules
        template_name, template_content = await self.get_template_for_context(
            base_template_name="planner_template", task=task, context=context
        )

        # Construct final prompt with all planning context
        planning_context["prompt"] = self._construct_prompt(
            template_name=template_name,
            task=task,
            planning_type=planning_context["planning_type"],
            scope=planning_context["scope"],
            complexity_level=planning_context["complexity_level"],
            requirements_analysis=planning_context["requirements_analysis"],
            dependencies_analysis=planning_context["dependencies_analysis"],
            risk_factors=planning_context["risk_factors"],
            effort_estimates=planning_context["effort_estimates"],
            quality_requirements=self._get_planning_quality_requirements(
                planning_context
            ),
        )

        return planning_context
    #
    # ========================================================================
    # Method 2.4: analyze_requirements_document
    # ========================================================================
    #
    async def _analyze_requirements_document(
        self, content: str, file_id: str, task: str
    ) -> Dict[str, Any]:
        """
        Analyze a requirements document to extract planning insights.

        Args:
            content: The content of the requirements document
            file_id: The file identifier
            task: The planning task

        Returns:
            Requirements analysis for planning
        """
        analysis = {
            "file_id": file_id,
            "document_type": self._detect_document_type(content),
            "functional_requirements": [],
            "non_functional_requirements": [],
            "constraints": [],
            "acceptance_criteria": [],
            "complexity_indicators": [],
            "estimated_scope": "unknown",
        }

        try:
            # Extract functional requirements
            functional_patterns = [
                r"(?:shall|must|will|should)\s+([^.]+)",
                r"(?:functional requirement|FR\d+)[:\-\s]*([^.\n]+)",
                r"(?:the system|application|software)\s+(?:shall|must|will)\s+([^.]+)",
            ]

            for pattern in functional_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                analysis["functional_requirements"].extend(
                    [match.strip() for match in matches]
                )

            # Extract non-functional requirements
            nfr_keywords = [
                "performance",
                "scalability",
                "security",
                "usability",
                "reliability",
                "maintainability",
            ]
            for keyword in nfr_keywords:
                pattern = rf"{keyword}[:\-\s]*([^.\n]+)"
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["non_functional_requirements"].extend(
                        [f"{keyword}: {match.strip()}" for match in matches]
                    )

            # Extract constraints
            constraint_patterns = [
                r"(?:constraint|limitation|restriction)[:\-\s]*([^.\n]+)",
                r"(?:must not|cannot|shall not)\s+([^.]+)",
                r"(?:within|by|before)\s+(\d+\s+(?:days?|weeks?|months?))",
            ]

            for pattern in constraint_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                analysis["constraints"].extend([match.strip() for match in matches])

            # Detect complexity indicators
            complexity_keywords = [
                "integration",
                "api",
                "database",
                "authentication",
                "workflow",
                "reporting",
                "analytics",
                "real-time",
                "scalable",
                "distributed",
            ]

            for keyword in complexity_keywords:
                if keyword.lower() in content.lower():
                    analysis["complexity_indicators"].append(keyword)

            # Estimate scope based on content length and complexity
            word_count = len(content.split())
            complexity_score = len(analysis["complexity_indicators"])

            if word_count > 2000 or complexity_score > 5:
                analysis["estimated_scope"] = "large"
            elif word_count > 500 or complexity_score > 2:
                analysis["estimated_scope"] = "medium"
            else:
                analysis["estimated_scope"] = "small"

        except Exception as e:
            logger.error(
                f"[{self.name}] Error analyzing requirements document {file_id}: {e}"
            )
            analysis["error"] = str(e)

        return analysis
    #
    # ========================================================================
    # Async Method 2.5: analyze_project_dependencies
    # ========================================================================
    #
    async def _analyze_project_dependencies(
        self, planning_context: Dict, task: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze project dependencies based on requirements and context.

        Args:
            planning_context: Planning context with requirements analysis
            task: The planning task

        Returns:
            List of identified dependencies
        """
        dependencies = []

        try:
            # Technical dependencies based on requirements
            technical_deps = []
            for req_analysis in planning_context.get("requirements_analysis", []):
                complexity_indicators = req_analysis.get("complexity_indicators", [])

                # Map complexity indicators to technical dependencies
                if "database" in complexity_indicators:
                    technical_deps.append(
                        {
                            "type": "technical",
                            "name": "Database Design & Setup",
                            "description": "Database schema design and infrastructure setup",
                            "criticality": "high",
                            "estimated_effort": "3-5 days",
                        }
                    )

                if "api" in complexity_indicators:
                    technical_deps.append(
                        {
                            "type": "technical",
                            "name": "API Design & Development",
                            "description": "REST/GraphQL API design and implementation",
                            "criticality": "high",
                            "estimated_effort": "2-4 days",
                        }
                    )

                if "authentication" in complexity_indicators:
                    technical_deps.append(
                        {
                            "type": "technical",
                            "name": "Authentication System",
                            "description": "User authentication and authorization implementation",
                            "criticality": "medium",
                            "estimated_effort": "2-3 days",
                        }
                    )

            dependencies.extend(technical_deps)

            # Resource dependencies based on team size and complexity
            team_size = planning_context.get("team_size", "small")
            if team_size in ["large", "enterprise"]:
                dependencies.append(
                    {
                        "type": "resource",
                        "name": "Team Coordination",
                        "description": "Inter-team communication and coordination setup",
                        "criticality": "medium",
                        "estimated_effort": "ongoing",
                    }
                )

            # Timeline dependencies
            timeline_constraint = planning_context.get(
                "timeline_constraint", "flexible"
            )
            if timeline_constraint == "strict":
                dependencies.append(
                    {
                        "type": "timeline",
                        "name": "Parallel Development Streams",
                        "description": "Setup parallel development to meet strict deadlines",
                        "criticality": "high",
                        "estimated_effort": "planning phase",
                    }
                )

        except Exception as e:
            logger.error(f"[{self.name}] Error analyzing project dependencies: {e}")

        return dependencies
    #
    # ========================================================================
    # Async Method 2.6: assess_project_risks
    # ========================================================================
    #
    async def _assess_project_risks(
        self, planning_context: Dict, task: str
    ) -> List[Dict[str, Any]]:
        """
        Assess project risks based on complexity, requirements, and constraints.

        Args:
            planning_context: Planning context with analysis results
            task: The planning task

        Returns:
            List of identified risk factors
        """
        risks = []

        try:
            # Technical risks based on complexity
            complexity_level = planning_context.get("complexity_level", "moderate")
            if complexity_level in ["high", "very_high"]:
                risks.append(
                    {
                        "type": "technical",
                        "name": "Technical Complexity Risk",
                        "description": "High technical complexity may lead to underestimation and delays",
                        "probability": "medium",
                        "impact": "high",
                        "mitigation": "Add 25% buffer to estimates, plan for technical spikes",
                    }
                )

            # Timeline risks
            timeline_constraint = planning_context.get(
                "timeline_constraint", "flexible"
            )
            if timeline_constraint == "strict":
                risks.append(
                    {
                        "type": "timeline",
                        "name": "Schedule Pressure Risk",
                        "description": "Strict timeline may compromise quality or scope",
                        "probability": "high",
                        "impact": "medium",
                        "mitigation": "Define clear scope boundaries, plan for scope reduction if needed",
                    }
                )

            # Integration risks
            for req_analysis in planning_context.get("requirements_analysis", []):
                complexity_indicators = req_analysis.get("complexity_indicators", [])
                if "integration" in complexity_indicators:
                    risks.append(
                        {
                            "type": "integration",
                            "name": "Third-party Integration Risk",
                            "description": "External system integration may introduce delays and compatibility issues",
                            "probability": "medium",
                            "impact": "medium",
                            "mitigation": "Plan integration testing early, have fallback options",
                        }
                    )

            # Resource risks
            team_size = planning_context.get("team_size", "small")
            if team_size == "small":
                risks.append(
                    {
                        "type": "resource",
                        "name": "Limited Resource Risk",
                        "description": "Small team may lack specialized skills or bandwidth",
                        "probability": "medium",
                        "impact": "medium",
                        "mitigation": "Cross-train team members, plan for external consultation if needed",
                    }
                )

        except Exception as e:
            logger.error(f"[{self.name}] Error assessing project risks: {e}")

        return risks
    #
    # ========================================================================
    # Async Method 2.7: generate_effort_estimates
    # ========================================================================
    #
    async def _generate_effort_estimates(
        self, planning_context: Dict, task: str
    ) -> Dict[str, Any]:
        """
        Generate effort estimates based on requirements analysis and complexity.

        Args:
            planning_context: Planning context with requirements and dependencies
            task: The planning task

        Returns:
            Effort estimation breakdown
        """
        estimates = {
            "methodology": self.default_estimation_method,
            "total_estimate": {"min": 0, "max": 0, "most_likely": 0},
            "phase_breakdown": {},
            "resource_allocation": {},
            "confidence_level": "medium",
        }

        try:
            # Base estimates by scope
            scope_estimates = {
                "small": {"min": 5, "max": 15, "most_likely": 10},
                "medium": {"min": 15, "max": 40, "most_likely": 25},
                "large": {"min": 40, "max": 100, "most_likely": 65},
                "enterprise": {"min": 100, "max": 250, "most_likely": 150},
            }

            # Get base estimate from scope
            estimated_scope = "medium"  # Default
            for req_analysis in planning_context.get("requirements_analysis", []):
                if req_analysis.get("estimated_scope") in scope_estimates:
                    estimated_scope = req_analysis["estimated_scope"]
                    break

            base_estimate = scope_estimates.get(
                estimated_scope, scope_estimates["medium"]
            )

            # Apply complexity multipliers
            complexity_level = planning_context.get("complexity_level", "moderate")
            complexity_multipliers = {
                "low": 0.8,
                "moderate": 1.0,
                "high": 1.3,
                "very_high": 1.6,
            }

            multiplier = complexity_multipliers.get(complexity_level, 1.0)

            # Calculate adjusted estimates (in person-days)
            estimates["total_estimate"] = {
                "min": int(base_estimate["min"] * multiplier),
                "max": int(base_estimate["max"] * multiplier),
                "most_likely": int(base_estimate["most_likely"] * multiplier),
            }

            # Phase breakdown (percentages)
            estimates["phase_breakdown"] = {
                "planning_analysis": {
                    "percentage": 10,
                    "days": int(estimates["total_estimate"]["most_likely"] * 0.10),
                },
                "design_architecture": {
                    "percentage": 15,
                    "days": int(estimates["total_estimate"]["most_likely"] * 0.15),
                },
                "development": {
                    "percentage": 50,
                    "days": int(estimates["total_estimate"]["most_likely"] * 0.50),
                },
                "testing_qa": {
                    "percentage": 20,
                    "days": int(estimates["total_estimate"]["most_likely"] * 0.20),
                },
                "deployment_support": {
                    "percentage": 5,
                    "days": int(estimates["total_estimate"]["most_likely"] * 0.05),
                },
            }

            # Resource allocation
            team_size = planning_context.get("team_size", "small")
            if team_size == "small":
                estimates["resource_allocation"] = {
                    "developer": {"count": 1, "percentage": 70},
                    "qa": {"count": 1, "percentage": 20},
                    "devops": {"count": 0.5, "percentage": 10},
                }
            elif team_size == "medium":
                estimates["resource_allocation"] = {
                    "developer": {"count": 2, "percentage": 60},
                    "qa": {"count": 1, "percentage": 25},
                    "devops": {"count": 0.5, "percentage": 10},
                    "design": {"count": 0.5, "percentage": 5},
                }
            else:  # large team
                estimates["resource_allocation"] = {
                    "developer": {"count": 3, "percentage": 55},
                    "qa": {"count": 1.5, "percentage": 25},
                    "devops": {"count": 1, "percentage": 15},
                    "design": {"count": 0.5, "percentage": 5},
                }

            # Confidence level based on requirements clarity
            requirements_count = sum(
                len(req_analysis.get("functional_requirements", []))
                for req_analysis in planning_context.get("requirements_analysis", [])
            )

            if requirements_count > 10:
                estimates["confidence_level"] = "high"
            elif requirements_count > 5:
                estimates["confidence_level"] = "medium"
            else:
                estimates["confidence_level"] = "low"

        except Exception as e:
            logger.error(f"[{self.name}] Error generating effort estimates: {e}")

        return estimates
    #
    # ========================================================================
    # Method 2.8: detect_document_type
    # ========================================================================
    #
    def _detect_document_type(self, content: str) -> str:
        """
        Detect the      type of planning document based on content patterns.

        Args:
            content: Document content to analyze

        Returns:
            Detected document type
        """
        content_lower = content.lower()

        # Check for specific document type indicators
        if any(
            keyword in content_lower
            for keyword in ["user story", "acceptance criteria", "as a user"]
        ):
            return "user_stories"
        elif any(
            keyword in content_lower
            for keyword in [
                "technical specification",
                "api specification",
                "system design",
            ]
        ):
            return "technical_spec"
        elif any(
            keyword in content_lower
            for keyword in ["business requirements", "business rules", "stakeholder"]
        ):
            return "business_requirements"
        elif any(
            keyword in content_lower
            for keyword in ["test plan", "test case", "test scenario"]
        ):
            return "test_plan"
        elif any(
            keyword in content_lower
            for keyword in ["project charter", "project plan", "milestone"]
        ):
            return "project_plan"
        else:
            return "general_requirements"
    #
    # ========================================================================
    # Method 2.9: _get_planning_quality_requirements
    # ========================================================================
    #
    def _get_planning_quality_requirements(
        self, planning_context: Dict
    ) -> Dict[str, Any]:
        """
        Determine quality requirements for planning based on context and agent configuration.

        Args:
            planning_context: Planning context dictionary

        Returns:
            Quality requirements specification
        """
        return {
            "min_sections": self.min_plan_sections,
            "require_timeline": self.require_timeline,
            "require_dependencies": self.require_dependencies,
            "require_risk_analysis": self.require_risk_analysis,
            "estimation_method": self.default_estimation_method,
            "complexity_factors": self.complexity_factors,
            "resource_types": self.resource_types,
            "planning_type": planning_context.get("planning_type", "comprehensive"),
        }
    #
    # ========================================================================
    # Async Method 2.10: _post_planning_processing
    # ========================================================================
    #
    async def _post_planning_processing(
        self, response: str, planning_context: Dict, task: str, context: Dict
    ) -> None:
        """
        Process generated planning output for quality assurance and file operations.

        Args:
            response: Generated planning content
            planning_context: Planning generation context
            task: Original task description
            context: Execution context
        """
        try:
            # Validate planning completeness
            quality_score = self._assess_planning_quality(response, planning_context)

            if quality_score < 70:
                logger.warning(
                    f"[{self.name}] Planning quality score: {quality_score}% - below threshold"
                )
            else:
                logger.info(f"[{self.name}] Planning quality score: {quality_score}%")

            # Generate planning metrics
            metrics = self._generate_planning_metrics(response, planning_context)
            logger.info(f"[{self.name}] Planning metrics: {metrics}")

            # Save planning output if file operations are requested
            if context.get("save_to_file") and context.get("output_file_id"):
                await self._save_planning_file(
                    response, context["output_file_id"], planning_context
                )

        except Exception as e:
            logger.error(f"[{self.name}] Error in post-planning processing: {e}")
    #
    # ========================================================================
    # Method 2.11: _assess_planning_quality
    # ========================================================================
    #
    def _assess_planning_quality(self, plan: str, planning_context: Dict) -> int:
        """
        Assess the quality of generated planning output against requirements.

        Args:
            plan: Generated planning content
            planning_context: Planning context with requirements

        Returns:
            Quality score (0-100)
        """
        score = 0
        total_checks = 0

        try:
            # Check for required planning sections
            required_sections = [
                "objectives",
                "scope",
                "timeline",
                "resources",
                "risks",
                "deliverables",
            ]
            sections_found = 0

            for section in required_sections:
                if section.lower() in plan.lower():
                    sections_found += 1

            if required_sections:
                score += (sections_found / len(required_sections)) * 25
                total_checks += 25

            # Check plan length (should be substantial)
            if len(plan) > 1000:
                score += 20
            elif len(plan) > 500:
                score += 10
            total_checks += 20

            # Check for effort estimates
            estimate_indicators = [
                "estimate",
                "effort",
                "duration",
                "timeline",
                "person-days",
            ]
            estimate_score = sum(
                3 for indicator in estimate_indicators if indicator in plan.lower()
            )
            score += min(15, estimate_score)
            total_checks += 15

            # Check for risk analysis
            risk_indicators = ["risk", "mitigation", "contingency", "assumption"]
            if any(indicator in plan.lower() for indicator in risk_indicators):
                score += 15
            total_checks += 15

            # Check for dependency analysis
            dependency_indicators = [
                "dependency",
                "prerequisite",
                "blocker",
                "requires",
            ]
            if any(indicator in plan.lower() for indicator in dependency_indicators):
                score += 15
            total_checks += 15

            # Check for resource planning
            resource_indicators = ["resource", "team", "role", "allocation"]
            if any(indicator in plan.lower() for indicator in resource_indicators):
                score += 10
            total_checks += 10

            # Normalize score
            if total_checks > 0:
                score = min(100, (score / total_checks) * 100)

        except Exception as e:
            logger.error(f"[{self.name}] Error assessing planning quality: {e}")
            score = 50  # Default middle score on error

        return int(score)
    #
    # ========================================================================
    # Method 2.12: _generate_planning_metrics
    # ========================================================================
    #
    def _generate_planning_metrics(
        self, plan: str, planning_context: Dict
    ) -> Dict[str, Any]:
        """
        Generate metrics for the planning output.

        Args:
            plan: Generated planning content
            planning_context: Planning context

        Returns:
            Planning metrics dictionary
        """
        try:
            metrics = {
                "plan_length": len(plan),
                "word_count": len(plan.split()),
                "sections_identified": 0,
                "estimates_provided": False,
                "risks_identified": 0,
                "dependencies_identified": 0,
            }

            # Count sections
            section_indicators = [
                "##",
                "###",
                "**",
                "objectives:",
                "scope:",
                "timeline:",
                "risks:",
            ]
            metrics["sections_identified"] = sum(
                1 for indicator in section_indicators if indicator in plan.lower()
            )

            # Check for estimates
            metrics["estimates_provided"] = any(
                word in plan.lower() for word in ["days", "weeks", "hours", "estimate"]
            )

            # Count risks
            metrics["risks_identified"] = plan.lower().count("risk")

            # Count dependencies
            metrics["dependencies_identified"] = plan.lower().count("depend")

            return metrics

        except Exception:
            return {"error": "Failed to generate metrics"}
    #
    # ========================================================================
    # Async Method 2.13: _save_planning_file
    # ========================================================================
    #
    async def _save_planning_file(
        self, content: str, file_id: str, planning_context: Dict
    ) -> None:
        """
        Save generated planning output to the specified file.

        Args:
            content: Planning content to save
            file_id: Target file ID for saving
            planning_context: Planning context
        """
        try:
            # Add planning metadata header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"# Project Planning Document\n"
            header += f"Generated: {timestamp}\n"
            header += f"Planning Type: {planning_context.get('planning_type', 'comprehensive')}\n"
            header += f"Scope: {planning_context.get('scope', 'feature')}\n"
            header += f"Complexity: {planning_context.get('complexity_level', 'moderate')}\n\n"
            header += "---\n\n"

            # Combine header with content
            final_content = header + content

            # Save using inherited GDrive functionality
            await self._save_gdrive_file(
                file_id, final_content, "Project Planning Document"
            )
            logger.info(f"[{self.name}] Planning document saved to file: {file_id}")

        except Exception as e:
            logger.error(f"[{self.name}] Error saving planning file: {e}")
    #
    # ========================================================================
    # Method 2.14: _calculate_timeline_estimates
    # ========================================================================
    #
    def _calculate_timeline_estimates(
        self, effort_estimates: Dict, resource_allocation: Dict
    ) -> Dict[str, Any]:
        """
        Calculate timeline estimates based on effort and resource allocation.

        Args:
            effort_estimates: Effort estimation data
            resource_allocation: Resource allocation plan

        Returns:
            Timeline calculation results
        """
        try:
            timeline = {
                "duration_days": 0,
                "duration_weeks": 0,
                "start_date": None,
                "end_date": None,
                "milestones": [],
            }

            # Calculate duration based on effort and team capacity
            total_effort = effort_estimates.get("total_estimate", {}).get(
                "most_likely", 30
            )
            total_resources = sum(
                res.get("count", 1) for res in resource_allocation.values()
            )

            # Assume 80% efficiency for team collaboration overhead
            efficiency_factor = 0.8 if total_resources > 1 else 1.0

            timeline["duration_days"] = int(
                total_effort / (total_resources * efficiency_factor)
            )
            timeline["duration_weeks"] = max(
                1, timeline["duration_days"] // 5
            )  # 5 working days per week

            # Generate milestone estimates
            phase_breakdown = effort_estimates.get("phase_breakdown", {})
            cumulative_days = 0

            for phase, details in phase_breakdown.items():
                phase_days = details.get("days", 0)
                cumulative_days += phase_days

                timeline["milestones"].append(
                    {
                        "phase": phase.replace("_", " ").title(),
                        "duration_days": phase_days,
                        "cumulative_days": cumulative_days,
                        "percentage_complete": int(
                            (cumulative_days / total_effort) * 100
                        ),
                    }
                )

            return timeline

        except Exception as e:
            logger.error(f"[{self.name}] Error calculating timeline estimates: {e}")
            return {"error": "Timeline calculation failed"}
    #
    # ========================================================================
    # Method 2.15: _validate_planning_constraints
    # ========================================================================
    #
    def _validate_planning_constraints(
        self, planning_context: Dict, estimates: Dict
    ) -> List[str]:
        """
        Validate planning output against constraints and business rules.

        Args:
            planning_context: Planning context with constraints
            estimates: Generated effort estimates

        Returns:
            List of constraint violations or warnings
        """
        violations = []

        try:
            # Check timeline constraints
            timeline_constraint = planning_context.get(
                "timeline_constraint", "flexible"
            )
            estimated_duration = estimates.get("total_estimate", {}).get(
                "most_likely", 0
            )

            if timeline_constraint == "strict" and estimated_duration > 30:
                violations.append(
                    "Estimated duration exceeds strict timeline constraint of 30 days"
                )
            elif timeline_constraint == "moderate" and estimated_duration > 60:
                violations.append(
                    "Estimated duration exceeds moderate timeline constraint of 60 days"
                )

            # Check resource constraints
            team_size = planning_context.get("team_size", "small")
            total_resources = sum(
                res.get("count", 1)
                for res in estimates.get("resource_allocation", {}).values()
            )

            max_team_sizes = {"small": 3, "medium": 6, "large": 12}
            max_size = max_team_sizes.get(team_size, 3)

            if total_resources > max_size:
                violations.append(
                    f"Resource allocation ({total_resources}) exceeds {team_size} team size limit ({max_size})"
                )

            # Check complexity vs. team capability
            complexity_level = planning_context.get("complexity_level", "moderate")
            if complexity_level in ["high", "very_high"] and team_size == "small":
                violations.append(
                    "High complexity project may require larger team than currently allocated"
                )

        except Exception as e:
            logger.error(f"[{self.name}] Error validating planning constraints: {e}")
            violations.append(f"Constraint validation error: {str(e)}")

        return violations
#
#
## End of planner_agent.py
