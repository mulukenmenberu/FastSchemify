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
            type: Database access type - 'orm' (SQLAlchemy ORM) or 'query' (raw SQL queries). Default: 'orm'
            output: Output directory for generated project. Default: 'generated_api'
            **kwargs: Additional configuration options (for future use)
        """
        self.type = type.lower()
        if self.type not in ["orm", "query"]:
            raise ValueError(
                f"Invalid type: '{type}'. Must be 'orm' (SQLAlchemy ORM) or 'query' (raw SQL queries). "
                f"Note: Use 'query' instead of 'sql' for raw SQL mode."
            )
        
        self.output_dir = Path(output)
        self.use_orm = self.type == "orm"
        self.kwargs = kwargs
        
        # Will be set during generation
        self.db_connection = None
        self.schema_discovery = None
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def connect(self):
        """Connect to database and discover schemas"""
        print("✓  Connecting to database...")
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
        self.use_orm = use_orm  # True for SQLAlchemy ORM, False for raw SQL queries
    
    def generate_project(self):
        """Generate complete FastAPI project"""
        if not self.schemas:
            raise ValueError("No schemas discovered. Connect to database first.")
        
        # Create output directory
        if self.output_dir.exists():
            print(f"✓  Output directory '{self.output_dir}' exists. Removing...")
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Creating project structure in '{self.output_dir}'...")
        
        # Show mode
        mode = "SQLAlchemy ORM" if self.use_orm else "Raw SQL Queries"
        print(f"✓ Using {mode} mode")
        
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
        """Generate SQLAlchemy ORM models with declarative classes"""
        # Generate database.py with Base and engine
        database_content = '''"""
Database configuration and base model
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import Optional
from app.core.config import settings

Base = declarative_base()

# Lazy-loaded engine
_engine = None
SessionLocal = None

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

def get_session():
    """Get database session"""
    global SessionLocal
    if SessionLocal is None:
        engine = get_engine()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

def init_tables():
    """Initialize database tables (for compatibility)"""
    # Import all models to register them with Base
    from app.models import database
    # Models are imported via __init__.py
    pass
'''
        (self.output_dir / "app" / "models" / "database.py").write_text(database_content)
        
        # Generate individual model files for each table
        model_imports = []
        for table_name, schema in self.schemas.items():
            model_class_name = self._to_class_name(table_name)
            model_content = self._generate_model_class(table_name, schema)
            model_file = self.output_dir / "app" / "models" / f"{table_name}.py"
            model_file.write_text(model_content)
            model_imports.append(f"from app.models.{table_name} import {model_class_name}")
        
        # Generate models __init__.py
        init_content = '''"""
Database models
"""
from app.models.database import Base

'''
        init_content += '\n'.join(model_imports)
        init_content += '\n\n__all__ = [\n'
        for table_name in self.schemas.keys():
            init_content += f'    "{self._to_class_name(table_name)}",\n'
        init_content += ']\n'
        
        (self.output_dir / "app" / "models" / "__init__.py").write_text(init_content)
    
    def _generate_model_class(self, table_name: str, schema: Dict[str, Any]) -> str:
        """Generate a SQLAlchemy declarative model class for a table"""
        class_name = self._to_class_name(table_name)
        columns = []
        import_types = set()
        
        # Get primary keys
        primary_keys = schema.get("primary_keys", [])
        if not primary_keys:
            # Try to find from columns
            for col_name, col_info in schema["columns"].items():
                if col_info.get("primary_key"):
                    primary_keys.append(col_name)
        
        # Generate columns
        for col_name, col_info in schema["columns"].items():
            col_def, col_imports = self._generate_column_definition(col_name, col_info, primary_keys, schema)
            columns.append(col_def)
            import_types.update(col_imports)
        
        # Build imports
        import_lines = ["from sqlalchemy import Column"]
        if import_types:
            type_imports = sorted(list(import_types))
            import_lines.append(f"from sqlalchemy import {', '.join(type_imports)}")
        
        # Generate foreign key relationships (simplified for now)
        relationships = []
        
        model_content = f'''"""
SQLAlchemy ORM model for {table_name} table
"""
{chr(10).join(import_lines)}
from app.models.database import Base


class {class_name}(Base):
    """ORM model for {table_name}"""
    __tablename__ = "{table_name}"
    
{chr(10).join(columns)}
'''
        if relationships:
            model_content += '\n' + '\n'.join(relationships) + '\n'
        
        return model_content
    
    def _generate_column_definition(self, col_name: str, col_info: Dict[str, Any], primary_keys: list, schema: Dict[str, Any]) -> tuple:
        """Generate SQLAlchemy column definition and return (definition, imports)"""
        sql_type = col_info["type"]
        nullable = col_info.get("nullable", False)
        is_pk = col_name in primary_keys or col_info.get("primary_key", False)
        default = col_info.get("default")
        
        # Map SQL types to SQLAlchemy types
        sa_type, imports = self._get_sqlalchemy_type(sql_type)
        
        # Build column definition
        parts = [f'    {col_name} = Column({sa_type}']
        
        # Add primary key
        if is_pk:
            parts.append("primary_key=True")
        
        # Add nullable
        if not nullable and not is_pk:
            parts.append("nullable=False")
        elif nullable and not is_pk:
            parts.append("nullable=True")
        
        # Add default
        if default is not None:
            if isinstance(default, str) and "CURRENT" in default.upper():
                if "DATE" in default.upper():
                    parts.append("server_default='CURRENT_DATE'")
                elif "TIME" in default.upper():
                    parts.append("server_default='CURRENT_TIMESTAMP'")
            else:
                # Try to evaluate default if it's a simple value
                try:
                    if default.isdigit():
                        parts.append(f"default={int(default)}")
                    else:
                        parts.append(f"default={repr(default)}")
                except:
                    parts.append(f"default={repr(default)}")
        
        # Auto-increment for integer primary keys
        if is_pk and "int" in sql_type.lower():
            parts.append("autoincrement=True")
        
        col_def = ", ".join(parts) + ")"
        return col_def, imports
    
    def _get_sqlalchemy_type(self, sql_type: str) -> tuple:
        """Convert SQL type to SQLAlchemy type, returns (type_string, imports_set)"""
        sql_type_lower = sql_type.lower()
        imports = set()
        
        if "int" in sql_type_lower:
            if "big" in sql_type_lower:
                imports.add("BigInteger")
                return "BigInteger", imports
            elif "small" in sql_type_lower:
                imports.add("SmallInteger")
                return "SmallInteger", imports
            else:
                imports.add("Integer")
                return "Integer", imports
        elif "float" in sql_type_lower or "double" in sql_type_lower:
            imports.add("Float")
            return "Float", imports
        elif "decimal" in sql_type_lower or "numeric" in sql_type_lower:
            imports.add("Numeric")
            return "Numeric", imports
        elif "bool" in sql_type_lower:
            imports.add("Boolean")
            return "Boolean", imports
        elif "date" in sql_type_lower and "time" not in sql_type_lower:
            imports.add("Date")
            return "Date", imports
        elif "time" in sql_type_lower or "timestamp" in sql_type_lower:
            imports.add("DateTime")
            return "DateTime", imports
        elif "text" in sql_type_lower:
            imports.add("Text")
            return "Text", imports
        elif "varchar" in sql_type_lower or "char" in sql_type_lower:
            imports.add("String")
            # Try to extract length
            import re
            match = re.search(r'\((\d+)\)', sql_type)
            if match:
                length = match.group(1)
                return f"String({length})", imports
            return "String(255)", imports
        else:
            imports.add("String")
            return "String(255)", imports  # Default
    
    def _to_class_name(self, table_name: str) -> str:
        """Convert table_name to PascalCase class name"""
        return ''.join(word.capitalize() for word in table_name.split('_'))
    
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
            
            # Handle relative paths - try multiple locations
            if not os.path.isabs(db_path):
                project_root = Path(__file__).parent.parent.parent
                
                # Try 1: Full path relative to project root
                full_path = project_root / db_path
                if full_path.exists():
                    db_path = str(full_path.absolute())
                else:
                    # Try 2: Just filename in project root
                    filename_only = Path(db_path).name
                    filename_path = project_root / filename_only
                    if filename_path.exists():
                        db_path = str(filename_path.absolute())
                    else:
                        # Try 3: Parent directory (where generator was run from)
                        parent_dir = project_root.parent
                        parent_path = parent_dir / db_path
                        if parent_path.exists():
                            db_path = str(parent_path.absolute())
                        else:
                            # Try 4: Parent directory with just filename
                            parent_filename = parent_dir / filename_only
                            if parent_filename.exists():
                                db_path = str(parent_filename.absolute())
                            else:
                                # Use project root as fallback
                                db_path = str(full_path.absolute())
            
            if not os.path.exists(db_path):
                raise FileNotFoundError(
                    f"SQLite database file not found: {db_path}. " +
                    "Please make sure the database file exists or update SQLITE_PATH in your .env file. " +
                    f"Tried: {db_path}"
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
        """Generate ORM-based services using SQLAlchemy ORM models"""
        for table_name, schema in self.schemas.items():
            primary_key = self.schema_discovery.get_primary_key(table_name)
            if not primary_key:
                primary_key = "id"  # Fallback
            
            model_class_name = self._to_class_name(table_name)
            service_class_name = f"{model_class_name}Service"
            
            service_content = f'''"""
Service layer for {table_name}
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from app.models.database import get_session
from app.models.{table_name} import {model_class_name}

class {service_class_name}:
    """Service for {table_name} operations"""
    
    @staticmethod
    async def get_all(skip: int = 0, limit: int = 100, sort_by: Optional[str] = None, order: str = "asc") -> List[Dict[str, Any]]:
        """Get all items"""
        db: Session = get_session()
        try:
            query = db.query({model_class_name})
            
            if sort_by:
                if order == "desc":
                    query = query.order_by(desc(getattr({model_class_name}, sort_by)))
                else:
                    query = query.order_by(getattr({model_class_name}, sort_by))
            
            items = query.offset(skip).limit(limit).all()
            return [{{k: v for k, v in item.__dict__.items() if not k.startswith('_')}} for item in items]
        finally:
            db.close()
    
    @staticmethod
    async def get_by_id(item_id: Any) -> Optional[Dict[str, Any]]:
        """Get item by ID"""
        db: Session = get_session()
        try:
            item = db.query({model_class_name}).filter(getattr({model_class_name}, "{primary_key}") == item_id).first()
            if not item:
                return None
            return {{k: v for k, v in item.__dict__.items() if not k.startswith('_')}}
        finally:
            db.close()
    
    @staticmethod
    async def create(item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item"""
        db: Session = get_session()
        try:
            item = {model_class_name}(**item_data)
            db.add(item)
            db.commit()
            db.refresh(item)
            return {{k: v for k, v in item.__dict__.items() if not k.startswith('_')}}
        finally:
            db.close()
    
    @staticmethod
    async def update(item_id: Any, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update item"""
        db: Session = get_session()
        try:
            item = db.query({model_class_name}).filter(getattr({model_class_name}, "{primary_key}") == item_id).first()
            if not item:
                return None
            
            for key, value in item_data.items():
                setattr(item, key, value)
            
            db.commit()
            db.refresh(item)
            return {{k: v for k, v in item.__dict__.items() if not k.startswith('_')}}
        finally:
            db.close()
    
    @staticmethod
    async def delete(item_id: Any) -> bool:
        """Delete item"""
        db: Session = get_session()
        try:
            item = db.query({model_class_name}).filter(getattr({model_class_name}, "{primary_key}") == item_id).first()
            if not item:
                return False
            db.delete(item)
            db.commit()
            return True
        finally:
            db.close()
    
    @staticmethod
    async def count() -> int:
        """Get total count"""
        db: Session = get_session()
        try:
            return db.query({model_class_name}).count()
        finally:
            db.close()
'''
            (self.output_dir / "app" / "services" / f"{table_name}_service.py").write_text(service_content)
    
    def _generate_sql_services(self):
        """Generate raw SQL-based services with proper parameterized queries"""
        for table_name, schema in self.schemas.items():
            primary_key = self.schema_discovery.get_primary_key(table_name)
            if not primary_key:
                primary_key = "id"  # Fallback
            
            # Get column names for SQL queries
            columns = list(schema["columns"].keys())
            columns_str = ", ".join(columns)
            service_class_name = f"{table_name.capitalize().replace('_', '')}Service"
            
            # Build column list for validation
            columns_list_str = str(columns)
            
            service_content = f'''"""
Service layer for {table_name} (Raw SQL)
"""
from typing import List, Dict, Any, Optional
from app.models.database import get_connection
from app.core.config import settings

class {service_class_name}:
    """Service for {table_name} operations using raw SQL"""
    
    @staticmethod
    async def get_all(skip: int = 0, limit: int = 100, sort_by: Optional[str] = None, order: str = "asc") -> List[Dict[str, Any]]:
        """Get all items"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            # Build query with column names (safe - from schema)
            query = "SELECT {columns_str} FROM {table_name}"
            
            # Add ORDER BY if specified (validate column name to prevent SQL injection)
            if sort_by:
                # Validate sort_by is a valid column name
                valid_columns = {columns_list_str}
                if sort_by not in valid_columns:
                    raise ValueError(f"Invalid sort column: {{sort_by}}")
                order_dir = "DESC" if order.lower() == "desc" else "ASC"
                query += f" ORDER BY {{sort_by}} {{order_dir}}"
            
            # Add LIMIT and OFFSET (parameterized)
            placeholder = "?" if settings.db_type == "sqlite" else "%s"
            query += f" LIMIT {{placeholder}} OFFSET {{placeholder}}"
            cursor.execute(query, (limit, skip))
            
            rows = cursor.fetchall()
            
            # Convert to dict
            if hasattr(cursor, "description") and cursor.description:
                column_names = [desc[0] for desc in cursor.description]
                return [dict(zip(column_names, row)) for row in rows]
            # For dict-like rows (sqlite3.Row, pymysql DictCursor, etc.)
            return [dict(row) for row in rows]
        finally:
            cursor.close()
    
    @staticmethod
    async def get_by_id(item_id: Any) -> Optional[Dict[str, Any]]:
        """Get item by ID"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            placeholder = "?" if settings.db_type == "sqlite" else "%s"
            query = "SELECT {columns_str} FROM {table_name} WHERE {primary_key} = " + placeholder
            cursor.execute(query, (item_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            if hasattr(cursor, "description") and cursor.description:
                column_names = [desc[0] for desc in cursor.description]
                return dict(zip(column_names, row))
            return dict(row)
        finally:
            cursor.close()
    
    @staticmethod
    async def create(item_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new item"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            # Remove primary key from insert if it's auto-increment and None
            insert_data = {{k: v for k, v in item_data.items() if k != "{primary_key}" or v is not None}}
            
            if not insert_data:
                raise ValueError("No data provided for insert")
            
            insert_columns = ", ".join(insert_data.keys())
            placeholder = "?" if settings.db_type == "sqlite" else "%s"
            insert_placeholders = ", ".join([placeholder] * len(insert_data))
            insert_values = list(insert_data.values())
            
            query = "INSERT INTO {table_name} (" + insert_columns + ") VALUES (" + insert_placeholders + ")"
            
            if settings.db_type in ["postgresql", "postgres"]:
                query += " RETURNING *"
            
            cursor.execute(query, insert_values)
            conn.commit()
            
            # Get the inserted row
            if settings.db_type in ["postgresql", "postgres"]:
                row = cursor.fetchone()
                if row:
                    if hasattr(cursor, "description") and cursor.description:
                        column_names = [desc[0] for desc in cursor.description]
                        return dict(zip(column_names, row))
                    return dict(row)
            elif settings.db_type == "mysql":
                # MySQL doesn't support RETURNING, fetch by last insert id
                if "{primary_key}" in item_data and item_data["{primary_key}"]:
                    return await {service_class_name}.get_by_id(item_data["{primary_key}"])
                else:
                    last_id = cursor.lastrowid
                    if last_id:
                        return await {service_class_name}.get_by_id(last_id)
            else:  # SQLite
                last_id = cursor.lastrowid
                if last_id:
                    return await {service_class_name}.get_by_id(last_id)
            
            # Fallback: return the data we inserted
            return item_data
        finally:
            cursor.close()
    
    @staticmethod
    async def update(item_id: Any, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Update item"""
        if not item_data:
            return await {service_class_name}.get_by_id(item_id)
        
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            # Build SET clause with parameterized values
            placeholder = "?" if settings.db_type == "sqlite" else "%s"
            valid_columns = {columns_list_str}
            set_parts = []
            values = []
            
            for col in item_data.keys():
                # Validate column name
                if col not in valid_columns:
                    raise ValueError(f"Invalid column name: {{col}}")
                set_parts.append(col + " = " + placeholder)
                values.append(item_data[col])
            
            set_clause = ", ".join(set_parts)
            values.append(item_id)  # Add item_id for WHERE clause
            
            query = "UPDATE {table_name} SET " + set_clause + " WHERE {primary_key} = " + placeholder
            
            if settings.db_type in ["postgresql", "postgres"]:
                query += " RETURNING *"
            
            cursor.execute(query, values)
            conn.commit()
            
            if settings.db_type in ["postgresql", "postgres"]:
                row = cursor.fetchone()
                if row:
                    if hasattr(cursor, "description") and cursor.description:
                        column_names = [desc[0] for desc in cursor.description]
                        return dict(zip(column_names, row))
                    return dict(row)
            
            if cursor.rowcount == 0:
                return None
            
            return await {service_class_name}.get_by_id(item_id)
        finally:
            cursor.close()
    
    @staticmethod
    async def delete(item_id: Any) -> bool:
        """Delete item"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            placeholder = "?" if settings.db_type == "sqlite" else "%s"
            query = "DELETE FROM {table_name} WHERE {primary_key} = " + placeholder
            cursor.execute(query, (item_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            cursor.close()
    
    @staticmethod
    async def count() -> int:
        """Get total count"""
        conn = get_connection()
        cursor = conn.cursor()
        
        try:
            query = "SELECT COUNT(*) FROM {table_name}"
            cursor.execute(query)
            
            result = cursor.fetchone()
            if result:
                if isinstance(result, dict):
                    # For dict cursors, get first value
                    return list(result.values())[0]
                elif isinstance(result, (list, tuple)):
                    return result[0]
                else:
                    return int(result)
            return 0
        finally:
            cursor.close()
'''
            (self.output_dir / "app" / "services" / f"{table_name}_service.py").write_text(service_content)
    
    def _generate_routers(self):
        """Generate FastAPI routers and Pydantic schemas"""
        # First, generate Pydantic schemas for each table
        for table_name, schema in self.schemas.items():
            self._generate_pydantic_schemas(table_name, schema)
        
        # Then generate routers that import the schemas
        for table_name, schema in self.schemas.items():
            primary_key = self.schema_discovery.get_primary_key(table_name)
            if not primary_key:
                primary_key = "id"
            
            service_name = f"{table_name.capitalize().replace('_', '')}Service"
            schema_class_prefix = table_name.capitalize().replace("_", "")
            
            router_content = f'''"""
API router for {table_name}
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from app.services.{table_name}_service import {service_name}
from app.schemas.{table_name} import {schema_class_prefix}Create, {schema_class_prefix}Update

router = APIRouter(prefix="/{table_name}", tags=["{table_name}"])


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
async def create_item(item: {schema_class_prefix}Create):
    """Create a new {table_name} item"""
    return await {service_name}.create(item.dict(exclude_unset=True))


@router.put("/{{item_id}}", response_model=dict)
async def update_item(item_id: int, item: {schema_class_prefix}Update):
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
    
    def _generate_pydantic_schemas(self, table_name: str, schema: Dict[str, Any]):
        """Generate Pydantic schemas for a table"""
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
        
        schema_class_prefix = table_name.capitalize().replace("_", "")
        
        schema_content = f'''"""
Pydantic schemas for {table_name}
"""
from typing import Optional
from pydantic import BaseModel


class {schema_class_prefix}Create(BaseModel):
    """Create schema for {table_name}"""
{chr(10).join(create_fields) if create_fields else "    pass"}

    class Config:
        from_attributes = True


class {schema_class_prefix}Update(BaseModel):
    """Update schema for {table_name}"""
{chr(10).join(update_fields) if update_fields else "    pass"}

    class Config:
        from_attributes = True
'''
        (self.output_dir / "app" / "schemas" / f"{table_name}.py").write_text(schema_content)
        
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
