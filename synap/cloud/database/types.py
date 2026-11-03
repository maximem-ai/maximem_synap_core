"""Database type utilities."""
from sqlalchemy import Enum as SQLEnum


def db_enum(enum_cls, name: str, **kwargs):
    """SQLAlchemy Enum that persists Python enum values (not names) to PostgreSQL.

    Use this for all str Enum columns. PostgreSQL migrations define lowercase
    values; Python enums use UPPERCASE names with matching lowercase values.
    Without values_callable, SQLAlchemy persists names, causing LookupError on read.
    """
    return SQLEnum(
        enum_cls,
        name=name,
        create_type=False,
        values_callable=lambda x: [e.value for e in x],
        **kwargs
    )
