"""Base repository with common CRUD operations.

Provides a generic repository pattern for SQLAlchemy ORM models.
"""

import logging
from typing import Generic, TypeVar, Type, Optional, List, Any
from uuid import UUID

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from synap.cloud.database.connection import Base

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """Base repository with common CRUD operations.
    
    Provides generic methods for create, read, update, delete operations
    that can be extended by specific repositories.
    
    Attributes:
        model: The SQLAlchemy ORM model class.
        session: The async database session.
    """
    
    model: Type[ModelT]
    
    def __init__(self, session: AsyncSession):
        """Initialize the repository with a database session.

        Args:
            session: Async SQLAlchemy session for database operations.
        """
        self.session = session
        self._logger = logger.getChild(self.model.__name__ if hasattr(self, 'model') else 'unknown')
    
    async def get_by_id(self, id: UUID | str) -> Optional[ModelT]:
        """Get a record by its primary key.

        Args:
            id: The primary key value (UUID or string).

        Returns:
            The model instance if found, None otherwise.
        """
        self._logger.debug(f"Fetching record by ID: id={id}")
        try:
            result = await self.session.execute(
                select(self.model).where(self.model.id == id)
            )
            record = result.scalar_one_or_none()
            if record:
                self._logger.debug(f"Record found: id={id}")
            else:
                self._logger.debug(f"Record not found: id={id}")
            return record
        except Exception as e:
            self._logger.exception(f"Database error during get_by_id: id={id}, error_type={type(e).__name__}")
            raise
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[ModelT]:
        """Get all records with pagination.
        
        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.
            
        Returns:
            List of model instances.
        """
        result = await self.session.execute(
            select(self.model).limit(limit).offset(offset)
        )
        return list(result.scalars().all())
    
    async def create(self, obj: ModelT) -> ModelT:
        """Create a new record.

        Args:
            obj: The model instance to persist.

        Returns:
            The persisted model instance with generated ID.
        """
        self._logger.info("Creating record")
        try:
            self.session.add(obj)
            await self.session.flush()
            await self.session.refresh(obj)
            self._logger.info(f"Record created successfully: id={obj.id if hasattr(obj, 'id') else 'N/A'}")
            return obj
        except Exception as e:
            self._logger.exception(f"Database error during create: error_type={type(e).__name__}")
            raise
    
    async def update(self, id: UUID | str, **kwargs: Any) -> Optional[ModelT]:
        """Update a record by ID.

        Args:
            id: The primary key value.
            **kwargs: Fields to update.

        Returns:
            The updated model instance if found, None otherwise.
        """
        self._logger.info(f"Updating record: id={id}")
        try:
            await self.session.execute(
                update(self.model)
                .where(self.model.id == id)
                .values(**kwargs)
            )
            result = await self.get_by_id(id)
            if result:
                self._logger.info(f"Record updated successfully: id={id}")
            else:
                self._logger.warning(f"Record not found for update: id={id}")
            return result
        except Exception as e:
            self._logger.exception(f"Database error during update: id={id}, error_type={type(e).__name__}")
            raise
    
    async def delete(self, id: UUID | str) -> bool:
        """Delete a record by ID.

        Args:
            id: The primary key value.

        Returns:
            True if a record was deleted, False otherwise.
        """
        self._logger.info(f"Deleting record: id={id}")
        try:
            result = await self.session.execute(
                delete(self.model).where(self.model.id == id)
            )
            deleted = result.rowcount > 0
            if deleted:
                self._logger.info(f"Record deleted successfully: id={id}")
            else:
                self._logger.warning(f"Record not found for deletion: id={id}")
            return deleted
        except Exception as e:
            self._logger.exception(f"Database error during delete: id={id}, error_type={type(e).__name__}")
            raise
    
    async def exists(self, id: UUID | str) -> bool:
        """Check if a record exists.
        
        Args:
            id: The primary key value.
            
        Returns:
            True if the record exists, False otherwise.
        """
        result = await self.session.execute(
            select(self.model.id).where(self.model.id == id)
        )
        return result.scalar_one_or_none() is not None
