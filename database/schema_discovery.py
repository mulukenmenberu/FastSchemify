"""
Schema discovery and inference module
"""
from typing import Dict, Any, List, Optional
from database.connections import DatabaseConnection, get_database_connection
from sqlalchemy import MetaData, Table, select
from sqlalchemy.engine import Engine
from pymongo import MongoClient


class SchemaDiscovery:
    """Discover and infer database schemas"""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def discover_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Discover schemas for all tables/collections"""
        tables = self.db_connection.get_tables()
        for table_name in tables:
            self.schemas[table_name] = self.db_connection.get_schema(table_name)
        return self.schemas
    
    def get_table_schema(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific table"""
        if table_name not in self.schemas:
            self.schemas[table_name] = self.db_connection.get_schema(table_name)
        return self.schemas.get(table_name)
    
    def get_primary_key(self, table_name: str) -> Optional[str]:
        """Get primary key column name for a table"""
        schema = self.get_table_schema(table_name)
        if schema and schema.get("primary_keys"):
            return schema["primary_keys"][0]
        return None
    
    def get_column_types(self, table_name: str) -> Dict[str, str]:
        """Get column types for a table"""
        schema = self.get_table_schema(table_name)
        if schema:
            return {
                col_name: col_info["type"]
                for col_name, col_info in schema.get("columns", {}).items()
            }
        return {}

