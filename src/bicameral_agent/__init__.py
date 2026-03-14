"""Bicameral agent framework for LLM episode tracking and evaluation."""

from bicameral_agent.schema import (
    ContextInjection,
    Episode,
    EpisodeOutcome,
    Message,
    ToolInvocation,
    UserEvent,
    UserEventType,
)
from bicameral_agent.serialization import (
    episode_from_parquet,
    episode_to_parquet,
    episodes_from_parquet,
    episodes_to_parquet,
)
from bicameral_agent.logger import ConversationLogger
from bicameral_agent.queue import ContextQueue, InterruptConfig, Priority, QueueItem, QueueState
from bicameral_agent.replay import DecisionPoint, EpisodeReplayer, ReplayState
from bicameral_agent.validation import EpisodeValidator, ValidationResult
from bicameral_agent.embeddings import Embedder, FastEmbedEmbedder, HashEmbedder
from bicameral_agent.encoder import FEATURE_DIM, StateEncoder
from bicameral_agent.followup_classifier import FollowUpClassifier, FollowUpType
from bicameral_agent.latency import APILatencyModel, LatencyEstimate
from bicameral_agent.token_estimator import ContextFeatures, TokenEstimate, TokenEstimator
from bicameral_agent.tool_latency import (
    CostEstimate,
    SubCallPrediction,
    ToolLatencyModel,
    ToolPrediction,
)
from bicameral_agent.conscious_loop import AssistantResponse, ConsciousLoop
from bicameral_agent.gemini import ChatMessage, GeminiClient, GeminiResponse
from bicameral_agent.dataset import (
    ResearchQADataset,
    ResearchQATask,
    TaskDifficulty,
    TaskSplit,
)
from bicameral_agent.scorer import LexicalScorer, TaskScore, TaskScorer
from bicameral_agent.simulated_user import (
    ActionType,
    Patience,
    SimulatedUser,
    Strictness,
    UserAction,
)
from bicameral_agent.assumption_auditor import (
    AssumptionAuditor,
    EvidenceResult,
    EvidenceVerdict,
    IdentifiedAssumption,
    RiskLevel,
    SuggestedAction,
)
from bicameral_agent.gap_scanner import (
    GapCategory,
    IdentifiedGap,
    MockSearchProvider,
    ResearchGapScanner,
    SearchResult,
)
from bicameral_agent.signal_classifier import (
    LengthRatio,
    ResponseLatency,
    SentimentShift,
    SIGNAL_DIM,
    SignalClassifier,
    SignalVector,
    StopCount,
)
from bicameral_agent.heuristic_controller import (
    Action,
    DecisionLog,
    ExecutingTool,
    FullState,
    HeuristicController,
)
from bicameral_agent.episode_runner import (
    Controller,
    EpisodeConfig,
    EpisodeRunner,
    InjectionMode,
)
from bicameral_agent.random_controller import RandomController
from bicameral_agent.coherence_judge import CoherenceJudge, CoherenceScore
from bicameral_agent.ab_test import (
    ABTestResult,
    ABTestRunner,
    Condition,
    default_conditions,
)
from bicameral_agent.tool_primitive import (
    BudgetExceededError,
    StateVector,
    TokenBudget,
    ToolMetadata,
    ToolPrimitive,
    ToolResult,
)

try:
    from bicameral_agent.policy_value_net import (
        ACTION_ORDER,
        NUM_ACTIONS,
        PolicyValueNetwork,
    )
except ImportError:  # torch not installed
    pass

__all__ = [
    "ACTION_ORDER",
    "ABTestResult",
    "ABTestRunner",
    "Action",
    "ActionType",
    "APILatencyModel",
    "AssistantResponse",
    "AssumptionAuditor",
    "BudgetExceededError",
    "ChatMessage",
    "CoherenceJudge",
    "CoherenceScore",
    "Condition",
    "ConsciousLoop",
    "ContextFeatures",
    "ContextInjection",
    "ContextQueue",
    "ConversationLogger",
    "CostEstimate",
    "DecisionLog",
    "Controller",
    "DecisionPoint",
    "ExecutingTool",
    "Embedder",
    "Episode",
    "EpisodeConfig",
    "EpisodeOutcome",
    "EpisodeReplayer",
    "EpisodeRunner",
    "EpisodeValidator",
    "EvidenceResult",
    "EvidenceVerdict",
    "FEATURE_DIM",
    "FastEmbedEmbedder",
    "FollowUpClassifier",
    "FollowUpType",
    "FullState",
    "GapCategory",
    "GeminiClient",
    "GeminiResponse",
    "HashEmbedder",
    "HeuristicController",
    "IdentifiedAssumption",
    "IdentifiedGap",
    "InjectionMode",
    "InterruptConfig",
    "LatencyEstimate",
    "LengthRatio",
    "LexicalScorer",
    "Message",
    "MockSearchProvider",
    "NUM_ACTIONS",
    "Patience",
    "PolicyValueNetwork",
    "Priority",
    "QueueItem",
    "QueueState",
    "RandomController",
    "ReplayState",
    "ResearchGapScanner",
    "ResearchQADataset",
    "ResearchQATask",
    "ResponseLatency",
    "RiskLevel",
    "SIGNAL_DIM",
    "SearchResult",
    "SentimentShift",
    "SimulatedUser",
    "SignalClassifier",
    "SignalVector",
    "StateEncoder",
    "StateVector",
    "Strictness",
    "StopCount",
    "SubCallPrediction",
    "SuggestedAction",
    "TaskDifficulty",
    "TaskScore",
    "TaskScorer",
    "TaskSplit",
    "TokenBudget",
    "TokenEstimate",
    "TokenEstimator",
    "ToolInvocation",
    "ToolLatencyModel",
    "ToolMetadata",
    "ToolPrediction",
    "ToolPrimitive",
    "ToolResult",
    "UserAction",
    "UserEvent",
    "UserEventType",
    "ValidationResult",
    "default_conditions",
    "episode_from_parquet",
    "episode_to_parquet",
    "episodes_from_parquet",
    "episodes_to_parquet",
]
