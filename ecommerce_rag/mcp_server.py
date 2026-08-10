"""MCP exposure for the guarded retail tool surface.

MCP is an integration boundary, not an alternate execution path. Every call is
delegated to :class:`RetailTools`, including identity and transactional guards.
The authenticated user id is server-side configuration and is never accepted
from an MCP tool argument.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from .tools import RetailTools


class MCPRetailFacade:
    def __init__(self, tools: RetailTools, user_id: str):
        self.tools = tools
        self.user_id = user_id

    def _user_call(self, name: str, **arguments: Any) -> dict[str, Any]:
        if not self.user_id:
            return {"ok": False, "changed": False, "error": "mcp_user_not_configured"}
        return self.tools.call(name, user_id=self.user_id, **arguments)

    def search_catalog(self, query: str, top_k: int = 5,
                       category: str | None = None, max_price: float | None = None) -> dict[str, Any]:
        """Search products with optional category and budget constraints."""
        return self.tools.call("search_catalog", query=query, top_k=top_k,
                               category=category, max_price=max_price)

    def get_product(self, product_id: str) -> dict[str, Any]:
        """Read one product card using an internal product id from search."""
        return self.tools.call("get_product", product_id=product_id)

    def compare_products(self, product_ids: list[str]) -> dict[str, Any]:
        """Compare two or more products by internal id."""
        return self.tools.call("compare_products", product_ids=product_ids)

    def get_policy(self, policy_type: str) -> dict[str, Any]:
        """Read return, warranty, shipping, invoice, or refund policy."""
        return self.tools.call("get_policy", policy_type=policy_type)

    def get_order(self, order_id: str, verification_code: str) -> dict[str, Any]:
        """Read an authenticated user's order after six-digit verification."""
        return self._user_call("get_order", order_id=order_id, verification_code=verification_code)

    def check_return_eligibility(self, order_id: str, verification_code: str) -> dict[str, Any]:
        """Check return eligibility without changing order state."""
        return self._user_call("check_return_eligibility", order_id=order_id,
                               verification_code=verification_code)

    def create_return_request(self, order_id: str, verification_code: str,
                              confirmed: bool) -> dict[str, Any]:
        """Create a return request only after eligibility and explicit confirmation."""
        return self._user_call("create_return_request", order_id=order_id,
                               verification_code=verification_code, confirmed=confirmed)

    def escalate_to_human(self, reason: str, order_id: str | None = None) -> dict[str, Any]:
        """Hand the authenticated user's conversation to a human agent."""
        return self._user_call("escalate_to_human", reason=reason, order_id=order_id)


def build_server(tools: RetailTools, user_id: str) -> FastMCP:
    facade = MCPRetailFacade(tools, user_id)
    server = FastMCP(
        "Trusted E-commerce Tools",
        instructions=(
            "Retail tools backed by the same identity, eligibility, confirmation, "
            "and transactional guardrails as the customer-support agent."
        ),
        stateless_http=True,
        json_response=True,
    )
    for function in (
        facade.search_catalog,
        facade.get_product,
        facade.compare_products,
        facade.get_policy,
        facade.get_order,
        facade.check_return_eligibility,
        facade.create_return_request,
        facade.escalate_to_human,
    ):
        server.tool()(function)
    return server


def _runtime_tools() -> RetailTools:
    db_path = Path(os.getenv("ERAG_MCP_DB", "logs/demo_agent.db"))
    index_path = os.getenv("ERAG_MCP_INDEX", "").strip()
    retriever = None
    if index_path:
        from .hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(Path(index_path))
    return RetailTools(db_path, retriever)


def main() -> None:
    server = build_server(_runtime_tools(), os.getenv("ERAG_MCP_USER_ID", ""))
    transport = os.getenv("ERAG_MCP_TRANSPORT", "stdio")
    if transport not in {"stdio", "streamable-http", "sse"}:
        raise ValueError("ERAG_MCP_TRANSPORT must be stdio, streamable-http, or sse")
    server.run(transport=transport)


if __name__ == "__main__":
    main()
