"""
AI agent implementation for the Walta Framework.
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from .security import WaltaCryptographicProvider

logger = logging.getLogger(__name__)

@dataclass
class AIConfig:
    """Configuration for the AI agent."""
    agent_id: str
    model_name: str = "gemini-pro"
    temperature: float = 0.7
    max_tokens: int = 1000
    vector_db_url: Optional[str] = None
    memory_limit: int = 1000
    safety_threshold: float = 0.95
    max_retry_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

class AgentError(Exception):
    """Custom exception for AI agent-related issues."""
    pass

class WaltaAIAgent:
    """
    AI agent for self-analysis, decision-making, and system improvement.
    """
    def __init__(
        self,
        config: AIConfig,
        security_provider: WaltaCryptographicProvider,
        logger: Optional[logging.Logger] = None
    ):
        self.config = config
        self.security = security_provider
        self.logger = logger or logging.getLogger(f"WaltaAI-{config.agent_id}")
        self.knowledge_base: Dict[str, Any] = {}
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.operation_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"AI Agent {self.config.agent_id} initialized with model {self.config.model_name}")

    async def perform_self_analysis(self, system_status: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze system status and provide recommendations.
        """
        try:
            start_time = time.time()
            
            # Record operation start
            operation = {
                "type": "self_analysis",
                "start_time": datetime.utcnow().isoformat(),
                "system_status": system_status
            }

            analysis_result = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.config.agent_id,
                "status_summary": "Operational",
                "recommendations": [],
                "metrics": {
                    "cpu_usage": system_status.get("cpu_usage", 0),
                    "memory_usage": system_status.get("memory_usage", 0),
                    "response_time": 0
                }
            }

            # Check initialization status
            if system_status.get("initialization_status") != "OPERATIONAL":
                analysis_result["status_summary"] = "System requires attention"
                analysis_result["recommendations"].append(
                    "Investigate initialization failures and verify system configuration"
                )

            # Check system metrics
            if system_status.get("memory_usage", 0) > 80:
                analysis_result["recommendations"].append(
                    "High memory usage detected. Consider optimizing resource allocation"
                )

            # Update metrics
            analysis_result["metrics"]["response_time"] = time.time() - start_time
            
            # Record operation completion
            operation["end_time"] = datetime.utcnow().isoformat()
            operation["result"] = "success"
            self.operation_history.append(operation)
            
            # Update last analysis
            self.last_analysis = analysis_result
            
            self.logger.info(
                f"Self-analysis completed. Generated {len(analysis_result['recommendations'])} recommendations"
            )
            return analysis_result

        except Exception as e:
            self.logger.error(f"Self-analysis failed: {e}")
            operation["end_time"] = datetime.utcnow().isoformat()
            operation["result"] = "error"
            operation["error"] = str(e)
            self.operation_history.append(operation)
            raise AgentError(f"Self-analysis failed: {e}")

    async def make_governance_decision(self, proposal: Dict[str, Any]) -> str:
        """
        Make decisions based on governance proposals.
        """
        try:
            start_time = time.time()
            
            # Record operation start
            operation = {
                "type": "governance_decision",
                "start_time": datetime.utcnow().isoformat(),
                "proposal": proposal
            }

            # Basic decision logic
            risk_level = float(proposal.get("risk_level", 0))
            impact_level = float(proposal.get("impact_level", 0))
            urgency_level = float(proposal.get("urgency_level", 0))
            
            # Calculate safety score
            safety_score = 1.0 - (risk_level / 10.0)
            
            # Make decision
            if safety_score < self.config.safety_threshold:
                decision = "REJECTED"
                reason = "Risk level exceeds safety threshold"
            elif risk_level > 7:
                decision = "REJECTED"
                reason = "High risk operation"
            else:
                decision = "APPROVED"
                reason = "Within acceptable risk parameters"

            # Record decision
            result = {
                "decision": decision,
                "reason": reason,
                "safety_score": safety_score,
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "response_time": time.time() - start_time,
                    "risk_level": risk_level,
                    "impact_level": impact_level,
                    "urgency_level": urgency_level
                }
            }

            # Record operation completion
            operation["end_time"] = datetime.utcnow().isoformat()
            operation["result"] = result
            self.operation_history.append(operation)

            self.logger.info(
                f"Governance decision made: {decision} (safety score: {safety_score:.2f})"
            )
            return decision

        except Exception as e:
            self.logger.error(f"Governance decision failed: {e}")
            operation["end_time"] = datetime.utcnow().isoformat()
            operation["result"] = "error"
            operation["error"] = str(e)
            self.operation_history.append(operation)
            raise AgentError(f"Governance decision failed: {e}")

    async def update_knowledge_base(self, new_data: Dict[str, Any]) -> None:
        """
        Update agent's knowledge base with new information.
        """
        try:
            # Validate and sanitize input
            if not isinstance(new_data, dict):
                raise ValueError("New data must be a dictionary")

            # Add timestamp to data
            new_data["_timestamp"] = datetime.utcnow().isoformat()
            
            # Update knowledge base
            self.knowledge_base.update(new_data)
            
            # Enforce memory limit
            if len(self.knowledge_base) > self.config.memory_limit:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.knowledge_base.keys(),
                    key=lambda k: self.knowledge_base[k].get("_timestamp", "")
                )
                for key in sorted_keys[:len(sorted_keys) - self.config.memory_limit]:
                    del self.knowledge_base[key]

            self.logger.debug(
                f"Knowledge base updated. Current size: {len(self.knowledge_base)} entries"
            )

        except Exception as e:
            self.logger.error(f"Failed to update knowledge base: {e}")
            raise AgentError(f"Knowledge base update failed: {e}")

    def get_operational_status(self) -> Dict[str, Any]:
        """
        Get current operational status of the agent.
        """
        return {
            "agent_id": self.config.agent_id,
            "model_name": self.config.model_name,
            "knowledge_base_size": len(self.knowledge_base),
            "last_analysis_time": self.last_analysis["timestamp"] if self.last_analysis else None,
            "operation_count": len(self.operation_history),
            "uptime": time.time() - self.operation_history[0]["start_time"] if self.operation_history else 0
        }
