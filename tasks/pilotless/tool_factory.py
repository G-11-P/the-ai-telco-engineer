# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tool factory for the pilotless OFDM receiver task.

Provides API documentation tools.
"""

from tool_lib.base import ToolProvider, EvalToolBase
from config import ToolsConfig
from tool_lib.sionna_doc import SionnaDoc
from langchain_core.tools import BaseTool


class ToolFactory(ToolProvider):
    """Factory for task-specific tools."""

    TOOL_TYPES = [SionnaDoc]

    def __init__(self, tools_config: ToolsConfig, *,
                 eval_tool: EvalToolBase, higher_is_better: bool):
        cfg = tools_config.sionna_doc_config
        planetary_cfg = cfg.get("planetary_6g_config", {})
        use_planetary_profile = planetary_cfg.get("enabled", False)

        retrieve_k = cfg["retrieve_k"]
        rerank_top_n = cfg["rerank_top_n"]
        cache_dir = cfg["cache_dir_path"]
        if use_planetary_profile:
            retrieve_k = planetary_cfg.get("retrieve_k", retrieve_k)
            rerank_top_n = planetary_cfg.get("rerank_top_n", rerank_top_n)
            cache_dir = planetary_cfg.get("cache_dir_path", cache_dir)

        self.sionna_doc = SionnaDoc(
            embedding_model=cfg["embedding_model"],
            embedding_base_url=cfg["embedding_base_url"],
            reranker_model=cfg["reranker_model"],
            reranker_base_url=cfg["reranker_base_url"],
            retrieve_k=retrieve_k,
            rerank_top_n=rerank_top_n,
            cache_dir=cache_dir,
        )
        self._workspace = None

    def get_tools(self) -> list[BaseTool]:
        return self.sionna_doc.get_tools()

    def set_workspace(self, workspace):
        self._workspace = workspace
        self.sionna_doc.set_workspace(workspace)
