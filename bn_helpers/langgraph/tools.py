from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


# INPUT SCHEMAS
class DConnectedInput(BaseModel):
    from_node: str = Field(description="Source node name")
    to_node: str = Field(description="Target node name")

class CommonCauseInput(BaseModel):
    node1: str = Field(description="First node name")
    node2: str = Field(description="Second node name")

class CommonEffectInput(BaseModel):
    node1: str = Field(description="First node name")
    node2: str = Field(description="Second node name")

class ProbNodeInput(BaseModel):
    node: str = Field(description="Node name to query probability")

class ProbNodeGivenEvidenceInput(BaseModel):
    node: str = Field(description="Target node to query probability")
    evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evidence dict like {'Smoking': 'smoker', 'Age': 'young'}"
    )

class HighestImpactInput(BaseModel):
    node: str = Field(description="Target node")
    evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evidence dict to compare impacts"
    )

class HighestImpactWithBackgroundInput(BaseModel):
    node: str = Field(description="Target node")
    new_evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="New evidence to compare"
    )
    background_evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Existing/background evidence already observed"
    )

class EvidenceChangeRelationshipInput(BaseModel):
    node1: str = Field(description="First node")
    node2: str = Field(description="Second node")
    evidence: List[str] = Field(description="List of evidence node names")

class EvidencesBlockInput(BaseModel):
    node1: str = Field(description="First node")
    node2: str = Field(description="Second node")

class ChildrenCheckInput(BaseModel):
    node: str = Field(description="Parent node to check")
    list_of_nodes: List[str] = Field(description="Nodes to check if they are children")


# TOOL FACTORY
def create_langchain_tools(net) -> List[StructuredTool]:
    """Create LangChain StructuredTool objects from existing BN tool functions."""
    from bn_helpers.tool_agent import (
        make_explain_d_connected_tool,
        make_explain_common_cause_tool,
        make_explain_common_effect_tool,
        get_prob_node_tool,
        get_prob_node_given_any_evidence_tool,
        get_highest_impact_evidence_contribute_to_node_tool,
        get_highest_impact_evidence_contribute_to_node_given_background_evidence_tool,
        check_evidences_change_relationship_between_two_nodes_tool,
        get_evidences_block_two_nodes_tool,
        check_if_evidences_children_of_node_tool,
    )

    bound_fns = {
        "check_d_connected": make_explain_d_connected_tool(net),
        "check_common_cause": make_explain_common_cause_tool(net),
        "check_common_effect": make_explain_common_effect_tool(net),
        "get_prob_node": get_prob_node_tool(net),
        "get_prob_node_given_any_evidence": get_prob_node_given_any_evidence_tool(net),
        "get_highest_impact_evidence": get_highest_impact_evidence_contribute_to_node_tool(net),
        "get_highest_impact_evidence_with_background": get_highest_impact_evidence_contribute_to_node_given_background_evidence_tool(net),
        "check_evidence_change_relationship": check_evidences_change_relationship_between_two_nodes_tool(net),
        "get_evidences_block": get_evidences_block_two_nodes_tool(net),
        "check_if_children": check_if_evidences_children_of_node_tool(net),
    }

    tools = [
        StructuredTool.from_function(
            func=bound_fns["check_d_connected"],
            name="check_d_connected",
            description=(
                "Check if two nodes are d-connected in the Bayesian network. "
                "D-connected means entering evidence for one node will change the probability of the other node. "
                "Use for: dependency, connection, influence, reachable, correlation queries."
            ),
            args_schema=DConnectedInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["check_common_cause"],
            name="check_common_cause",
            description=(
                "Find common causes (shared parent/ancestor nodes) between two nodes. "
                "Use for: common cause, shared ancestor, upstream cause, root cause queries."
            ),
            args_schema=CommonCauseInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["check_common_effect"],
            name="check_common_effect",
            description=(
                "Find common effects (shared child/descendant nodes) between two nodes. "
                "Use for: common effect, collider, shared outcome, downstream effect queries."
            ),
            args_schema=CommonEffectInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["get_prob_node"],
            name="get_prob_node",
            description=(
                "Get the marginal probability distribution of a node (without any evidence). "
                "Use for: probability, likelihood, chance, prior belief queries."
            ),
            args_schema=ProbNodeInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["get_prob_node_given_any_evidence"],
            name="get_prob_node_given_any_evidence",
            description=(
                "Get probability distribution of a node given specified evidence. "
                "Evidence is a dict mapping node names to observed states. "
                "Use for: conditional probability, posterior, belief update queries."
            ),
            args_schema=ProbNodeGivenEvidenceInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["get_highest_impact_evidence"],
            name="get_highest_impact_evidence",
            description=(
                "Find which evidence has the highest impact on a target node. "
                "Compares effects of each piece of evidence individually. "
                "Use for: most influential, biggest effect, strongest influence queries."
            ),
            args_schema=HighestImpactInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["get_highest_impact_evidence_with_background"],
            name="get_highest_impact_evidence_with_background",
            description=(
                "With existing background evidence, find which new evidence has highest impact on a node. "
                "Use for: incremental evidence impact, additional evidence effect queries."
            ),
            args_schema=HighestImpactWithBackgroundInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["check_evidence_change_relationship"],
            name="check_evidence_change_relationship",
            description=(
                "Check if observing certain evidence changes the dependency relationship between two nodes. "
                "Use for: conditioning effect, blocking, opening path queries."
            ),
            args_schema=EvidenceChangeRelationshipInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["get_evidences_block"],
            name="get_evidences_block",
            description=(
                "Find evidence that would block the dependency path between two nodes. "
                "Use for: d-separation, conditioning set, separator queries."
            ),
            args_schema=EvidencesBlockInput,
        ),
        StructuredTool.from_function(
            func=bound_fns["check_if_children"],
            name="check_if_children",
            description=(
                "Check if a list of nodes are children of a given node. "
                "Use for: parent-child relationship, direct descendants queries."
            ),
            args_schema=ChildrenCheckInput,
        ),
    ]

    return tools


def get_tools_by_name(tools: List[StructuredTool]) -> Dict[str, StructuredTool]:
    """Create a name -> tool mapping for quick lookup."""
    return {tool.name: tool for tool in tools}
