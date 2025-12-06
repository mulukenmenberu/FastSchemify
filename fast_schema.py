"""
FastSchema - FastAPI Schema Generator
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from database.connections import get_database_connection
from database.schema_discovery import SchemaDiscovery
from config import settings


class FastSchema:
    """Main class for generating FastAPI projects from database schemas"""
    
    def __init__(
        self,
        type: str = "orm",
        output: str = "generated_api",
        **kwargs
    ):
        """
        Initialize FastSchema generator
        
        Args:
            type: Database access type - 'orm' (SQLAlchemy) or 'sql' (raw SQL). Default: 'orm'
            output: Output directory for generated project. Default: 'generated_api'
            **kwargs: Additional configuration options (for future use)
        """
        self.type = type.lower()
        if self.type not in ["orm", "sql"]:
            raise ValueError(f"Invalid type: {self.type}. Must be 'orm' or 'sql'")
        
        self.output_dir = Path(output)
        self.use_orm = self.type == "orm"
        self.kwargs = kwargs
        
        # Will be set during generation
        self.db_connection = None
        self.schema_discovery = None
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def connect(self):
        """Connect to database and discover schemas"""
        print("🔌 Connecting to database...")
        self.db_connection = get_database_connection()
        
        if not self.db_connection:
            raise ValueError("Database connection not configured")
        
        if not self.db_connection.connect():
            raise ConnectionError("Failed to connect to database")
        
        print(f"✓ Connected to {settings.db_type} database")
        
        self.schema_discovery = SchemaDiscovery(self.db_connection)
        self.schemas = self.schema_discovery.discover_all_schemas()
        print(f"✓ Discovered {len(self.schemas)} tables/collections")
    
    def generate(self):
        """Generate the FastAPI project"""
        if not self.schemas:
            raise ValueError("No schemas discovered. Call connect() first.")
        
        generator = ProjectGenerator(
            output_dir=str(self.output_dir),
            use_orm=self.use_orm
        )
        
        # Set the discovered schemas and connection
        generator.db_connection = self.db_connection
        generator.schema_discovery = self.schema_discovery
        generator.schemas = self.schemas
        
        # Generate the project
        generator.generate_project()
    
    def run(self):
        """Connect and generate in one call"""
        self.connect()
        self.generate()

class ProjectGenerator:
    """Generate complete FastAPI project from database schema"""
    
    def __init__(self, output_dir: str = "generated_api", use_orm: bool = True):
        self.output_dir = Path(output_dir)
        self.db_connection = None
        self.schema_discovery = None
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.use_orm = use_orm  # True for SQLAlchemy ORM, False for raw SQL
    
    def generate_project(self):
        """Generate complete FastAPI project"""
        if not self.schemas:
            raise ValueError("No schemas discovered. Connect to database first.")
        
        # Create output directory
        if self.output_dir.exists():
            print(f"⚠️  Output directory '{self.output_dir}' exists. Removing...")
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Creating project structure in '{self.output_dir}'...")
        
        # Show mode
        mode = "SQLAlchemy ORM" if self.use_orm else "Raw SQL"
        print(f"🔧 Using {mode} mode")
        
        # Generate project structure
        self._create_directory_structure()
        self._generate_config()
        self._generate_models()
        self._generate_services()
        self._generate_routers()
        self._generate_main_app()
        self._generate_requirements()
        self._generate_readme()
        self._generate_env_example()
        self._copy_env_file()
        
        print(f"\n✅ Project generated successfully in '{self.output_dir}'")
        print(f"\n📝 Next steps:")
        print(f"   1. cd {self.output_dir}")
        print(f"   2. Copy your .env file or create one")
        print(f"   3. pip install -r requirements.txt")
        print(f"   4. python app.py")
    
    def _create_directory_structure(self):
        """Create project directory structure"""
        dirs = [
            "app",
            "app/api",
            "app/api/v1",
            "app/api/v1/endpoints",
            "app/core",
            "app/models",
            "app/schemas",
            "app/services",
            "app/db",
        ]
        for dir_path in dirs:
            (self.output_dir / dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py files
        for dir_path in dirs:
            init_file = self.output_dir / dir_path / "__init__.py"
            init_file.write_text('"""Generated module"""\n')
    
    def _generate_config(self):
        """Generate configuration file"""
        config_content = '''"""
Application configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings"""
    
    # Database connection settings
    db_type: Optional[str] = None
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_uri: Optional[str] = None
    sqlite_path: Optional[str] = None
    
    # API settings
    api_title: str = "Generated API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
'''
        (self.output_dir / "app" / "core" / "config.py").write_text(config_content)
    
    def _generate_models(self):
        """Generate database models (ORM or raw SQL)"""
        if self.use_orm:
            self._generate_orm_models()
        else:
            self._generate_sql_models()
    
    def _generate_orm_models(self):
        """Generate SQLAlchemy ORM models"""
        models_content = '''"""
SQLAlchemy database models
"""
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import declarative_base
from typing import Optional, Dict
from app.core.config import settings

Base = declarative_base()
metadata = MetaData()

# Lazy-loaded engine
_engine = None

def get_engine():
    """Get database engine (lazy initialization)"""
    global _engine
    if _engine is None:
        if not settings.db_type:
            raise ValueError("Database type not configured. Please set DB_TYPE in .env file")
        
        if settings.db_type == "sqlite":
            import os
            from pathlib import Path
            
            db_path = settings.sqlite_path or settings.db_name or "database.db"
            
            # Handle relative paths - resolve relative to project root (where .env is)
            if not os.path.isabs(db_path):
                # Get project root (where app.py and .env are located)
                # __file__ is in app/models/database.py, so go up 2 levels
                project_root = Path(__file__).parent.parent.parent
                full_path = project_root / db_path
                
                # Check if file exists
                if full_path.exists():
                    db_path = str(full_path.absolute())
                else:
                    # Try just the filename in project root
                    filename_only = Path(db_path).name
                    filename_path = project_root / filename_only
                    if filename_path.exists():
                        db_path = str(filename_path.absolute())
                    else:
                        # Use the path as-is, SQLAlchemy will try to create it
                        db_path = str(full_path.absolute())
            
            # Verify database file exists (for SQLite)
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    "SQLite database file not found: " + str(db_path) + 
                    ". Please make sure the database file exists or update SQLITE_PATH in your .env file."
                )
            
            connection_string = f"sqlite:///{db_path}"
        elif settings.db_type == "mysql":
            connection_string = (
                f"mysql+pymysql://{settings.db_user}:{settings.db_password}"
                f"@{settings.db_host}:{settings.db_port or 3306}/{settings.db_name}"
            )
        elif settings.db_type in ["postgresql", "postgres"]:
            connection_string = (
                f"postgresql+psycopg2://{settings.db_user}:{settings.db_password}"
                f"@{settings.db_host}:{settings.db_port or 5432}/{settings.db_name}"
            )
        else:
            raise ValueError(f"Unsupported database type: {settings.db_type}")
        
        _engine = create_engine(connection_string, echo=False)
    return _engine

# Lazy-loaded table references
_tables: Dict[str, Table] = {}

def get_table(table_name: str) -> Table:
    """Get table reference (lazy loading)"""
    if table_name not in _tables:
        engine = get_engine()
        try:
            _tables[table_name] = Table(table_name, metadata, autoload_with=engine)
        except Exception as e:
            # Provide better error message
            import os
            db_path = settings.sqlite_path or settings.db_name or "database.db"
            if settings.db_type == "sqlite":
                resolved_path = os.path.abspath(db_path) if not os.path.isabs(db_path) else db_path
                error_msg = (
                    "Table '" + str(table_name) + "' not found in database. " +
                    "Database path: " + str(db_path) + " (resolved: " + str(resolved_path) + "). " +
                    "Make sure the database file exists and contains the table. " +
                    "Original error: " + str(e)
                )
                raise ValueError(error_msg) from e
            raise
    return _tables[table_name]

def get_tables() -> Dict[str, Table]:
    """Get all table references"""
    return _tables

# Convenience accessor (for backward compatibility)
# Note: Use get_engine() and get_table() functions instead
tables = _tables
'''
        
        # Add table initialization function
        table_names = list(self.schemas.keys())
        models_content += f'\n\ndef init_tables():\n'
        models_content += f'    """Initialize all tables"""\n'
        for table_name in table_names:
            models_content += f'    get_table("{table_name}")\n'
        
        (self.output_dir / "app" / "models" / "database.py").write_text(models_content)
    
    def _generate_sql_models(self):
        """Generate raw SQL connection models"""
        models_content = '''"""
Raw SQL database connection
"""
from typing import Optional, Dict, Any
from app.core.config import settings
import os
from pathlib import Path

# Database connection
_connection = None

def get_connection():
    """Get database connection (lazy initialization)"""
    global _connection
    if _connection is None:
        if not settings.db_type:
            raise ValueError("Database type not configured. Please set DB_TYPE in .env file")
        
        if settings.db_type == "sqlite":
            import sqlite3
            db_path = settings.sqlite_path or settings.db_name or "database.db"
            
            # Handle relative paths
            if not os.path.isabs(db_path):
                project_root = Path(__file__).parent.parent.parent
                full_path = project_root / db_path
                if full_path.exists():
                    db_path = str(full_path.absolute())
                else:
                    filename_only = Path(db_path).name
                    filename_path = project_root / filename_only
                    if filename_path.exists():
                        db_path = str(filename_path.absolute())
                    else:
                        db_path = str(full_path.absolute())
            
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    "SQLite database file not found: " + str(db_path) + 
                    ". Please make sure the database file exists or update SQLITE_PATH in your .env file."
                )
            
            _connection = sqlite3.connect(db_path, check_same_thread=False)
            _connection.row_factory = sqlite3.Row
            
        elif settings.db_type == "mysql":
            import pymysql
            _connection = pymysql.connect(
                host=settings.db_host or "localhost",
                port=settings.db_port or 3306,
                user=settings.db_user,
                password=settings.db_password,
                database=settings.db_name,
                cursorclass=pymysql.cursors.DictCursor
            )
            
        elif settings.db_type in ["postgresql", "postgres"]:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            _connection = psycopg2.connect(
                host=settings.db_host or "localhost",
                port=settings.db_port or 5432,
                user=settings.db_user,
                password=settings.db_password,
                database=settings.db_name,
                cursor_factory=RealDictCursor
            )
        else:
            raise ValueError(f"Unsupported database type: {settings.db_type}")
    
    return _connection

def get_cursor():
    """Get database cursor"""
    return get_connection().cursor()

def init_tables():
    """Initialize database connection"""
    get_connection()
'''
        
        (self.output_dir / "app" / "models" / "database.py").write_text(models_content)
    
    def _generate_services(self):
        """Generate service layer for each table"""
        if self.use_orm:
            self._generate_orm_services()
        else:
            self._generate_sql_services()
    
    def _generate_orm_services(self):
        """Generate ORM-based services"""
        for table_name, schema in self.schemas.items():
            primary_key = self.schema_discovery.get_primary_key(table_name)
            if not primary_key:
                primary_key = "id"  # Fallback
            
            service_content = f'''"""
Service layer for {table_name}
"""
from typing import List, Dict, Any, Optional
from sqlalchemy import select, insert, update, delete, text
from app.models.database import get_engine, get_table

class {table_name.capitalize().replace("_", "")}Service:
    """Service for {table_name} operations"""
    
    @staticmethod
    async def get_all(skip: int = 0, limit: int = 100, sort_by: Optional[str] = None, order: str = "asc") -> List[Dict[str, Any]]:
        """Get all items"""
        table = get_table("{table_name}")
        query = select(table).offset(skip).limit(limit)
        
        if sort_by:
            if order == "desc":
                query = query.order_by(table.c[sort_by].desc())
            else:
                query = query.order_by(table.c[sort_by])
        
        with get_engine().connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()
            columns = result.keys()
            return [dict(zip(columns, row)) for row in rows]
    
    @staticmethod
    async def get_by_id(item_id: Any) -> Optional[Dict[str, Any]]:
        """Get item by ID"""
        table = get_table("{table_name}")
        query = select(table).where(table.c["{primary_key}"] == item_id)
        
        with get_engine().connect() as conn:
            result = conn.execute(query)
            row = result.fetchone()
            if not row:
                return None
            columns = result.keys()
            return dict(zip(columns, row))
    
    @staticmethod
    async def create(item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item"""
        table = get_table("{table_name}")
        query = insert(table).values(**item_data).returning(table)
        
        with get_engine().connect() as conn:
            result = conn.execute(query)
            conn.commit()
            row = result.fetchone()
            columns = result.keys()
            return dict(zip(columns, row))
    
    @staticmethod
    async def update(item_id: Any, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update item"""
        table = get_table("{table_name}")
        query = update(table).where(table.c["{primary_key}"] == item_id).values(**item_data).returning(table)
        
        with get_engine().connect() as conn:
            result = conn.execute(query)
            conn.commit()
            row = result.fetchone()
            if not row:
                return None
            columns = result.keys()
            return dict(zip(columns, row))
    
    @staticmethod
    async def delete(item_id: Any) -> bool:
        """Delete item"""
        table = get_table("{table_name}")
        query = delete(table).where(table.c["{primary_key}"] == item_id)
        
        with get_engine().connect() as conn:
            result = conn.execute(query)
            conn.commit()
            return result.rowcount > 0
    
    @staticmethod
    async def count() -> int:
        """Get total count"""
        table = get_table("{table_name}")
        query = select(text("COUNT(*)")).select_from(table)
        
        with get_engine().connect() as conn:
            result = conn.execute(query)
            return result.scalar()
'''
            service_file = table_name.capitalize().replace("_", "")
            (self.output_dir / "app" / "services" / f"{table_name}_service.py").write_text(service_content)
    
    def _generate_sql_services(self):
        """Generate raw SQL-based services"""
        for table_name, schema in self.schemas.items():
            primary_key = self.schema_discovery.get_primary_key(table_name)
            if not primary_key:
                primary_key = "id"  # Fallback
            
            # Get column names for SQL queries
            columns = list(schema["columns"].keys())
            columns_str = ", ".join(columns)
            
            service_content = f'''"""
Service layer for {table_name} (Raw SQL)
"""
from typing import List, Dict, Any, Optional
from app.models.database import get_connection, get_cursor
from app.core.config import settings

class {table_name.capitalize().replace("_", "")}Service:
    """Service for {table_name} operations using raw SQL"""
    
    @staticmethod
    async def get_all(skip: int = 0, limit: int = 100, sort_by: Optional[str] = None, order: str = "asc") -> List[Dict[str, Any]]:
        """Get all items"""
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "SELECT {columns_str} FROM {table_name}"
        
        if sort_by:
            order_dir = "DESC" if order == "desc" else "ASC"
            query += " ORDER BY " + sort_by + " " + order_dir
        
        # Add LIMIT and OFFSET
        query += " LIMIT " + str(limit) + " OFFSET " + str(skip)
        
        cursor.execute(query)
        
        rows = cursor.fetchall()
        if hasattr(cursor, "description") and cursor.description:
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        return [dict(row) for row in rows]
    
    @staticmethod
    async def get_by_id(item_id: Any) -> Optional[Dict[str, Any]]:
        """Get item by ID"""
        conn = get_connection()
        cursor = conn.cursor()
        
        placeholder = "?" if settings.db_type == "sqlite" else "%s"
        query = "SELECT {columns_str} FROM {table_name} WHERE {primary_key} = " + placeholder
        cursor.execute(query.format(columns_str=columns_str, table_name=table_name, primary_key=primary_key), (item_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        if hasattr(cursor, "description") and cursor.description:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return dict(row)
    
    @staticmethod
    async def create(item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item"""
        conn = get_connection()
        cursor = conn.cursor()
        
        # Remove primary key from insert if it's auto-increment
        insert_data = {{k: v for k, v in item_data.items() if k != "{primary_key}" or v is not None}}
        insert_columns = ", ".join(insert_data.keys())
        placeholder = "?" if settings.db_type == "sqlite" else "%s"
        insert_placeholders = ", ".join([placeholder] * len(insert_data))
        insert_values = list(insert_data.values())
        
        query = "INSERT INTO {table_name} (" + insert_columns + ") VALUES (" + insert_placeholders + ")"
        
        if settings.db_type in ["postgresql", "postgres"]:
            query += " RETURNING *"
        
        cursor.execute(query.format(table_name=table_name), insert_values)
        
        conn.commit()
        
        # Get the inserted row
        if settings.db_type in ["postgresql", "postgres"]:
            row = cursor.fetchone()
            if row:
                if hasattr(cursor, "description") and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return dict(row)
        elif settings.db_type == "mysql":
            # MySQL doesn't support RETURNING, fetch by last insert id
            if "{primary_key}" in item_data:
                return await {table_name.capitalize().replace("_", "")}Service.get_by_id(item_data["{primary_key}"])
            else:
                last_id = cursor.lastrowid
                return await {table_name.capitalize().replace("_", "")}Service.get_by_id(last_id)
        else:  # SQLite
            last_id = cursor.lastrowid
            return await {table_name.capitalize().replace("_", "")}Service.get_by_id(last_id)
        
        return item_data
    
    @staticmethod
    async def update(item_id: Any, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update item"""
        if not item_data:
            return await {table_name.capitalize().replace("_", "")}Service.get_by_id(item_id)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build SET clause
        placeholder = "?" if settings.db_type == "sqlite" else "%s"
        set_clause = ", ".join([col + " = " + placeholder for col in item_data.keys()])
        values = list(item_data.values()) + [item_id]
        
        query = "UPDATE {table_name} SET " + set_clause + " WHERE {primary_key} = " + placeholder
        
        if settings.db_type in ["postgresql", "postgres"]:
            query += " RETURNING *"
        
        cursor.execute(query.format(table_name=table_name, primary_key=primary_key), values)
        
        conn.commit()
        
        if settings.db_type in ["postgresql", "postgres"]:
            row = cursor.fetchone()
            if row:
                if hasattr(cursor, "description") and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, row))
                return dict(row)
        
        if cursor.rowcount == 0:
            return None
        
        return await {table_name.capitalize().replace("_", "")}Service.get_by_id(item_id)
    
    @staticmethod
    async def delete(item_id: Any) -> bool:
        """Delete item"""
        conn = get_connection()
        cursor = conn.cursor()
        
        placeholder = "?" if settings.db_type == "sqlite" else "%s"
        query = "DELETE FROM {table_name} WHERE {primary_key} = " + placeholder
        cursor.execute(query.format(table_name=table_name, primary_key=primary_key), (item_id,))
        
        conn.commit()
        return cursor.rowcount > 0
    
    @staticmethod
    async def count() -> int:
        """Get total count"""
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "SELECT COUNT(*) FROM {table_name}"
        cursor.execute(query.format(table_name=table_name))
        
        result = cursor.fetchone()
        if result:
            return result[0] if isinstance(result, (list, tuple)) else result
        return 0
'''
            (self.output_dir / "app" / "services" / f"{table_name}_service.py").write_text(service_content)
    
    def _generate_routers(self):
        """Generate FastAPI routers"""
        # Generate router for each table
        for table_name, schema in self.schemas.items():
            primary_key = self.schema_discovery.get_primary_key(table_name)
            if not primary_key:
                primary_key = "id"
            
            service_name = f"{table_name.capitalize().replace('_', '')}Service"
            
            # Generate Pydantic schemas
            create_fields = []
            update_fields = []
            
            for col_name, col_info in schema["columns"].items():
                if col_info.get("primary_key"):
                    continue
                
                col_type = self._get_pydantic_type(col_info["type"])
                nullable = col_info.get("nullable", False)
                
                if nullable:
                    create_fields.append(f'    {col_name}: Optional[{col_type}] = None')
                else:
                    create_fields.append(f'    {col_name}: {col_type}')
                
                update_fields.append(f'    {col_name}: Optional[{col_type}] = None')
            
            router_content = f'''"""
API router for {table_name}
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from app.services.{table_name}_service import {service_name}

router = APIRouter(prefix="/{table_name}", tags=["{table_name}"])


# Pydantic schemas
class {table_name.capitalize().replace("_", "")}Create(BaseModel):
    """Create schema for {table_name}"""
{chr(10).join(create_fields) if create_fields else "    pass"}

    class Config:
        from_attributes = True


class {table_name.capitalize().replace("_", "")}Update(BaseModel):
    """Update schema for {table_name}"""
{chr(10).join(update_fields) if update_fields else "    pass"}

    class Config:
        from_attributes = True


@router.get("/", response_model=List[dict])
async def get_all(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    sort_by: Optional[str] = Query(None),
    order: str = Query("asc", regex="^(asc|desc)$")
):
    """Get all {table_name} items"""
    return await {service_name}.get_all(skip=skip, limit=limit, sort_by=sort_by, order=order)


@router.get("/{{item_id}}", response_model=dict)
async def get_one(item_id: int):
    """Get a single {table_name} item by ID"""
    item = await {service_name}.get_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="{table_name.capitalize()} not found")
    return item


@router.post("/", response_model=dict, status_code=201)
async def create_item(item: {table_name.capitalize().replace("_", "")}Create):
    """Create a new {table_name} item"""
    return await {service_name}.create(item.dict(exclude_unset=True))


@router.put("/{{item_id}}", response_model=dict)
async def update_item(item_id: int, item: {table_name.capitalize().replace("_", "")}Update):
    """Update a {table_name} item"""
    result = await {service_name}.update(item_id, item.dict(exclude_unset=True))
    if not result:
        raise HTTPException(status_code=404, detail="{table_name.capitalize()} not found")
    return result


@router.patch("/{{item_id}}", response_model=dict)
async def patch_item(item_id: int, item: dict = Body(...)):
    """Partially update a {table_name} item"""
    result = await {service_name}.update(item_id, item)
    if not result:
        raise HTTPException(status_code=404, detail="{table_name.capitalize()} not found")
    return result


@router.delete("/{{item_id}}", status_code=204)
async def delete_item(item_id: int):
    """Delete a {table_name} item"""
    success = await {service_name}.delete(item_id)
    if not success:
        raise HTTPException(status_code=404, detail="{table_name.capitalize()} not found")
    return None


@router.get("/count/total", response_model=dict)
async def get_count():
    """Get total count of {table_name} items"""
    return {{"count": await {service_name}.count()}}
'''
            (self.output_dir / "app" / "api" / "v1" / "endpoints" / f"{table_name}.py").write_text(router_content)
        
        # Generate router index
        router_index = '''"""
API router index
"""
from fastapi import APIRouter
'''
        for table_name in self.schemas.keys():
            router_index += f'from app.api.v1.endpoints import {table_name} as {table_name}_router\n'
        
        router_index += '\napi_router = APIRouter()\n'
        for table_name in self.schemas.keys():
            router_index += f'api_router.include_router({table_name}_router.router)\n'
        
        (self.output_dir / "app" / "api" / "v1" / "__init__.py").write_text(router_index)
    
    def _generate_main_app(self):
        """Generate main app.py file"""
        app_content = '''"""
Main FastAPI application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1 import api_router
from app.models.database import init_tables

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="FastAPI application generated from database schema"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        init_tables()
        print("✓ Database tables initialized")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize tables: {e}")


# Include API routers
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FastAPI API",
        "version": settings.api_version,
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
'''
        (self.output_dir / "app" / "main.py").write_text(app_content)
        
        # Also create app.py in root for easier running
        root_app = '''"""
Main entry point - runs the FastAPI application
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
'''
        (self.output_dir / "app.py").write_text(root_app)
    
    def _generate_requirements(self):
        """Generate requirements.txt"""
        requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
pymysql==1.1.0
psycopg2-binary==2.9.9
python-multipart==0.0.6
'''
        (self.output_dir / "requirements.txt").write_text(requirements)
    
    def _generate_readme(self):
        """Generate README"""
        readme = f'''# Generated FastAPI API

This project was auto-generated from your database schema.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your database configuration (see `.env.example`).

## Running

```bash
python app.py
```

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Available Endpoints

'''
        for table_name in self.schemas.keys():
            readme += f'- `/api/v1/{table_name}/` - CRUD operations for {table_name}\n'
        
        (self.output_dir / "README.md").write_text(readme)
    
    def _generate_env_example(self):
        """Generate .env.example"""
        env_example = f'''# Database Configuration
DB_TYPE={settings.db_type}
'''
        if settings.db_type == "sqlite":
            env_example += f'SQLITE_PATH={settings.sqlite_path or settings.db_name or "database.db"}\n'
        else:
            env_example += f'''DB_HOST={settings.db_host or "localhost"}
DB_PORT={settings.db_port or (3306 if settings.db_type == "mysql" else 5432)}
DB_NAME={settings.db_name}
DB_USER={settings.db_user}
DB_PASSWORD={settings.db_password}
'''
        env_example += '''
# API Configuration
API_TITLE=Generated API
API_VERSION=1.0.0
API_PREFIX=/api/v1
'''
        (self.output_dir / ".env.example").write_text(env_example)
    
    def _copy_env_file(self):
        """Copy .env file to generated project if it exists"""
        import shutil
        import os
        
        env_file = Path(".env")
        if env_file.exists():
            shutil.copy(env_file, self.output_dir / ".env")
            print("✓ Copied .env file to generated project")
        else:
            # Create a basic .env file from current settings
            env_content = f'''# Database Configuration
DB_TYPE={settings.db_type}
'''
            if settings.db_type == "sqlite":
                db_path = settings.sqlite_path or settings.db_name or "database.db"
                # If database file exists, copy it to generated project
                if os.path.exists(db_path):
                    db_filename = os.path.basename(db_path)
                    shutil.copy(db_path, self.output_dir / db_filename)
                    env_content += f'SQLITE_PATH={db_filename}\n'
                    print(f"✓ Copied database file '{db_filename}' to generated project")
                else:
                    # Use relative path - will be resolved at runtime
                    env_content += f'SQLITE_PATH={db_path}\n'
            else:
                env_content += f'''DB_HOST={settings.db_host or "localhost"}
DB_PORT={settings.db_port or (3306 if settings.db_type == "mysql" else 5432)}
DB_NAME={settings.db_name}
DB_USER={settings.db_user}
DB_PASSWORD={settings.db_password}
'''
            env_content += '''
# API Configuration
API_TITLE=Generated API
API_VERSION=1.0.0
API_PREFIX=/api/v1
'''
            (self.output_dir / ".env").write_text(env_content)
            print("✓ Created .env file in generated project")
    
    def _get_pydantic_type(self, sql_type: str) -> str:
        """Convert SQL type to Pydantic type"""
        sql_type_lower = sql_type.lower()
        if "int" in sql_type_lower:
            return "int"
        elif "float" in sql_type_lower or "double" in sql_type_lower or "decimal" in sql_type_lower:
            return "float"
        elif "bool" in sql_type_lower:
            return "bool"
        else:
            return "str"
