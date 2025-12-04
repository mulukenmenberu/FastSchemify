"""
Automatic API endpoint generators
"""
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, create_model
from sqlalchemy import MetaData, Table, select, insert, update, delete, text
from sqlalchemy.engine import Engine
from pymongo import MongoClient
from database.connections import DatabaseConnection, MongoDBConnection
from database.schema_discovery import SchemaDiscovery
from config import settings
import json


class APIGenerator:
    """Generate REST API endpoints for database tables/collections"""
    
    def __init__(self, db_connection: DatabaseConnection, schema_discovery: SchemaDiscovery):
        self.db_connection = db_connection
        self.schema_discovery = schema_discovery
        self.routers: Dict[str, APIRouter] = {}
    
    def generate_pydantic_model(self, table_name: str, operation: str = "create") -> BaseModel:
        """Generate Pydantic model from schema"""
        schema = self.schema_discovery.get_table_schema(table_name)
        if not schema:
            raise ValueError(f"Schema not found for table: {table_name}")
        
        fields = {}
        for col_name, col_info in schema["columns"].items():
            # Skip primary key for create operations
            if operation == "create" and col_info.get("primary_key"):
                continue
            
            # Skip primary key for update operations (can't be updated)
            if operation == "update" and col_info.get("primary_key"):
                continue
            
            # Determine field type
            col_type = col_info["type"].lower()
            if "int" in col_type:
                field_type = int
            elif "float" in col_type or "double" in col_type or "decimal" in col_type:
                field_type = float
            elif "bool" in col_type:
                field_type = bool
            elif "date" in col_type or "time" in col_type:
                field_type = str
            else:
                field_type = str
            
            # For update operations, all fields should be optional (partial updates)
            if operation == "update":
                field_type = Optional[field_type]
                fields[col_name] = (field_type, None)
            else:
                # For create operations, make optional if nullable
                if col_info.get("nullable"):
                    field_type = Optional[field_type]
                fields[col_name] = (field_type, ... if not col_info.get("nullable") else None)
        
        model_name = f"{table_name.capitalize()}{operation.capitalize()}Model"
        return create_model(model_name, **fields)
    
    def generate_router(self, table_name: str) -> APIRouter:
        """Generate FastAPI router for a table"""
        router = APIRouter(prefix=f"/{table_name}", tags=[table_name])
        schema = self.schema_discovery.get_table_schema(table_name)
        primary_key = self.schema_discovery.get_primary_key(table_name)
        
        # Generate Pydantic models
        CreateModel = self.generate_pydantic_model(table_name, "create")
        UpdateModel = self.generate_pydantic_model(table_name, "update")
        
        # GET all items
        @router.get("/", response_model=List[Dict[str, Any]])
        async def get_all(
            skip: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=1000),
            sort_by: Optional[str] = Query(None),
            order: str = Query("asc", regex="^(asc|desc)$")
        ):
            """Get all items from the table"""
            return await self._get_all_items(table_name, skip, limit, sort_by, order)
        
        # GET single item by ID
        @router.get("/{item_id}", response_model=Dict[str, Any])
        async def get_one(item_id: Any):
            """Get a single item by ID"""
            return await self._get_item_by_id(table_name, item_id, primary_key)
        
        # POST create item
        @router.post("/", response_model=Dict[str, Any], status_code=201)
        async def create_item(item: CreateModel):
            """Create a new item"""
            return await self._create_item(table_name, item.dict(exclude_unset=True))
        
        # PUT update item
        @router.put("/{item_id}", response_model=Dict[str, Any])
        async def update_item(item_id: Any, item: UpdateModel):
            """Update an item by ID"""
            return await self._update_item(table_name, item_id, primary_key, item.dict(exclude_unset=True))
        
        # PATCH partial update
        @router.patch("/{item_id}", response_model=Dict[str, Any])
        async def patch_item(item_id: Any, item: Dict[str, Any] = Body(...)):
            """Partially update an item by ID"""
            return await self._update_item(table_name, item_id, primary_key, item)
        
        # DELETE item
        @router.delete("/{item_id}", status_code=204)
        async def delete_item(item_id: Any):
            """Delete an item by ID"""
            await self._delete_item(table_name, item_id, primary_key)
            return None
        
        # GET count
        @router.get("/count/total", response_model=Dict[str, int])
        async def get_count():
            """Get total count of items"""
            return {"count": await self._get_count(table_name)}
        
        return router
    
    async def _get_all_items(self, table_name: str, skip: int, limit: int, sort_by: Optional[str], order: str) -> List[Dict[str, Any]]:
        """Get all items from table"""
        if isinstance(self.db_connection, (MongoDBConnection,)):
            # MongoDB
            db = self.db_connection.client[settings.db_name]
            collection = db[table_name]
            query = {}
            cursor = collection.find(query).skip(skip).limit(limit)
            if sort_by:
                cursor = cursor.sort(sort_by, 1 if order == "asc" else -1)
            items = list(cursor)
            # Convert ObjectId to string
            for item in items:
                if "_id" in item:
                    item["_id"] = str(item["_id"])
            return items
        else:
            # SQL databases
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.db_connection.engine)
            query = select(table).offset(skip).limit(limit)
            if sort_by:
                if order == "desc":
                    query = query.order_by(table.c[sort_by].desc())
                else:
                    query = query.order_by(table.c[sort_by])
            
            with self.db_connection.engine.connect() as conn:
                result = conn.execute(query)
                rows = result.fetchall()
                columns = result.keys()
                return [dict(zip(columns, row)) for row in rows]
    
    async def _get_item_by_id(self, table_name: str, item_id: Any, primary_key: Optional[str]) -> Dict[str, Any]:
        """Get item by ID"""
        if not primary_key:
            raise HTTPException(status_code=400, detail="Primary key not found")
        
        if isinstance(self.db_connection, (MongoDBConnection,)):
            # MongoDB
            from bson import ObjectId
            db = self.db_connection.client[settings.db_name]
            collection = db[table_name]
            try:
                item = collection.find_one({"_id": ObjectId(item_id)})
            except:
                item = collection.find_one({"_id": item_id})
            
            if not item:
                raise HTTPException(status_code=404, detail="Item not found")
            
            item["_id"] = str(item["_id"])
            return item
        else:
            # SQL databases
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.db_connection.engine)
            query = select(table).where(table.c[primary_key] == item_id)
            
            with self.db_connection.engine.connect() as conn:
                result = conn.execute(query)
                row = result.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Item not found")
                columns = result.keys()
                return dict(zip(columns, row))
    
    async def _create_item(self, table_name: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item"""
        if isinstance(self.db_connection, (MongoDBConnection,)):
            # MongoDB
            db = self.db_connection.client[settings.db_name]
            collection = db[table_name]
            result = collection.insert_one(item_data)
            item_data["_id"] = str(result.inserted_id)
            return item_data
        else:
            # SQL databases
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.db_connection.engine)
            query = insert(table).values(**item_data).returning(table)
            
            with self.db_connection.engine.connect() as conn:
                result = conn.execute(query)
                conn.commit()
                row = result.fetchone()
                columns = result.keys()
                return dict(zip(columns, row))
    
    async def _update_item(self, table_name: str, item_id: Any, primary_key: Optional[str], item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update item"""
        if not primary_key:
            raise HTTPException(status_code=400, detail="Primary key not found")
        
        if isinstance(self.db_connection, (MongoDBConnection,)):
            # MongoDB
            from bson import ObjectId
            db = self.db_connection.client[settings.db_name]
            collection = db[table_name]
            try:
                result = collection.update_one({"_id": ObjectId(item_id)}, {"$set": item_data})
            except:
                result = collection.update_one({"_id": item_id}, {"$set": item_data})
            
            if result.matched_count == 0:
                raise HTTPException(status_code=404, detail="Item not found")
            
            return await self._get_item_by_id(table_name, item_id, primary_key)
        else:
            # SQL databases
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.db_connection.engine)
            query = update(table).where(table.c[primary_key] == item_id).values(**item_data).returning(table)
            
            with self.db_connection.engine.connect() as conn:
                result = conn.execute(query)
                conn.commit()
                row = result.fetchone()
                if not row:
                    raise HTTPException(status_code=404, detail="Item not found")
                columns = result.keys()
                return dict(zip(columns, row))
    
    async def _delete_item(self, table_name: str, item_id: Any, primary_key: Optional[str]):
        """Delete item"""
        if not primary_key:
            raise HTTPException(status_code=400, detail="Primary key not found")
        
        if isinstance(self.db_connection, (MongoDBConnection,)):
            # MongoDB
            from bson import ObjectId
            db = self.db_connection.client[settings.db_name]
            collection = db[table_name]
            try:
                result = collection.delete_one({"_id": ObjectId(item_id)})
            except:
                result = collection.delete_one({"_id": item_id})
            
            if result.deleted_count == 0:
                raise HTTPException(status_code=404, detail="Item not found")
        else:
            # SQL databases
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.db_connection.engine)
            query = delete(table).where(table.c[primary_key] == item_id)
            
            with self.db_connection.engine.connect() as conn:
                result = conn.execute(query)
                conn.commit()
                if result.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Item not found")
    
    async def _get_count(self, table_name: str) -> int:
        """Get total count of items"""
        if isinstance(self.db_connection, (MongoDBConnection,)):
            # MongoDB
            db = self.db_connection.client[settings.db_name]
            collection = db[table_name]
            return collection.count_documents({})
        else:
            # SQL databases
            metadata = MetaData()
            table = Table(table_name, metadata, autoload_with=self.db_connection.engine)
            query = select(text("COUNT(*)")).select_from(table)
            
            with self.db_connection.engine.connect() as conn:
                result = conn.execute(query)
                return result.scalar()

