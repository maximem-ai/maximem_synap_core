"""Shared AuthContext contract for cross-layer communication.

This contract defines the authentication context structure used across
Gateway, Core Services, and Agents. It is the immutable identity envelope
that flows through the system.

Contract stability: STABLE
Breaking changes require major version bump.

Version history:
- v1.0: Initial version with core identity fields
- v1.1: Added dashboard-specific fields (user_id, user_role, impersonation)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class AuthContext:
    """Immutable identity envelope for authenticated principals.
    
    This is the ONLY authentication context consumed by downstream services.
    All authentication and authorization decisions must be completed before
    this context is created.
    
    Attributes:
        client_id: Unique identifier for the client application
        org_id: Organization identifier
        instance_id: Instance identifier for the SDK/Agent (optional for dashboard)
        principal_type: Type of principal ("sdk" | "dashboard" | "agent")
        credential_type: Type of credential used ("api_key" | "mtls" | "jwt")
        scopes: List of authorized scopes/permissions
        issued_at: When this context was issued
        expires_at: When this context expires
        
        Dashboard-specific fields (when principal_type == "dashboard"):
        user_id: Dashboard user UUID
        user_role: User's role ("owner" | "admin" | "member" | "synap_admin" | etc.)
        
        Impersonation fields (when admin is impersonating):
        is_impersonating: Whether this is an impersonation session
        impersonation_session_id: UUID of the impersonation log entry
        actual_admin_user_id: The real admin user performing impersonation
        is_impersonation_read_only: Whether impersonation is read-only
    """
    
    # Core identity (required)
    client_id: str
    org_id: str
    principal_type: str          # "sdk" | "dashboard" | "agent"
    credential_type: str         # "api_key" | "mtls" | "jwt"
    scopes: List[str]
    issued_at: datetime
    expires_at: datetime
    
    # Instance ID (optional - not required for dashboard principals)
    instance_id: Optional[str] = None
    
    # Dashboard-specific fields (optional, used when principal_type == "dashboard")
    user_id: Optional[str] = None
    user_role: Optional[str] = None  # "owner" | "admin" | "member" | "synap_admin" | etc.
    
    # Impersonation fields (optional, used when admin is impersonating a client)
    is_impersonating: bool = False
    impersonation_session_id: Optional[str] = None
    actual_admin_user_id: Optional[str] = None
    is_impersonation_read_only: bool = True
    
    def __post_init__(self):
        """Validate context after initialization."""
        if self.scopes is None:
            self.scopes = []
    
    def can_write(self) -> bool:
        """Check if context allows write operations.
        
        Returns:
            True if write operations are allowed, False otherwise.
        """
        if self.is_impersonating and self.is_impersonation_read_only:
            return False
        return True
    
    def is_synap_admin(self) -> bool:
        """Check if user is a Synap admin.
        
        Returns:
            True if user has any Synap admin role.
        """
        return self.user_role in ("synap_admin", "synap_support", "synap_viewer")
    
    def is_client_admin(self) -> bool:
        """Check if user is a client admin or owner.
        
        Returns:
            True if user is an owner or admin of a client.
        """
        return self.user_role in ("owner", "admin")
    
    def is_dashboard_principal(self) -> bool:
        """Check if this is a dashboard principal.
        
        Returns:
            True if principal_type is "dashboard".
        """
        return self.principal_type == "dashboard"
    
    def get_effective_client_id(self) -> str:
        """Get the effective client ID for operations.
        
        During impersonation, returns the impersonated client's ID.
        Otherwise returns the normal client_id.
        
        Returns:
            The client ID to use for operations.
        """
        return self.client_id