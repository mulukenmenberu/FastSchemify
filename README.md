# FastAPI Schema Generator

> Automatically generate production-ready FastAPI REST APIs from your existing database schema. Zero boilerplate, maximum productivity.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

##  Overview

FastAPI Schema Generator is a powerful tool that automatically generates complete, production-ready FastAPI applications from your database. Simply connect to your database, and get a fully functional REST API with CRUD operations, automatic schema validation, and OpenAPI documentation.

### Key Features

- ⚡ **Zero Configuration** - Connect to your database and generate APIs instantly
- ⚡ **Multi-Database Support** - MySQL, PostgreSQL, MongoDB, and SQLite
- ⚡ **Automatic Schema Discovery** - Intelligently infers schemas from your database
- ⚡ **Full CRUD Operations** - Create, Read, Update, Delete endpoints for every table
- ⚡ **Auto-Generated Documentation** - Swagger/OpenAPI docs out of the box
- ⚡ **Production-Ready Architecture** - Clean separation of concerns with services, models, and routers
- 🔍 **Advanced Querying** - Built-in pagination, sorting, and filtering
- ⚡ **Type-Safe** - Full type hints and Pydantic validation

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fastapi-schema-generator.git
cd fastapi-schema-generator

# Install dependencies
pip install -r requirements.txt
```

### Generate Your First API

1. **Configure your database** by creating a `.env` file:

```env
DB_TYPE=sqlite
SQLITE_PATH=./database.db
```

2. **Generate the FastAPI project**:

```bash
python generate.py
```

3. **Run the generated API**:

```bash
cd generated_api
pip install -r requirements.txt
python app.py
```

4. **Access your API**:
   - API: http://localhost:8000
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## 📖 Documentation

### Database Configuration

#### SQLite

```env
DB_TYPE=sqlite
SQLITE_PATH=./database.db
```

#### MySQL

```env
DB_TYPE=mysql
DB_HOST=localhost
DB_PORT=3306
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

#### PostgreSQL

```env
DB_TYPE=postgresql
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password
```

#### MongoDB

```env
DB_TYPE=mongodb
DB_HOST=localhost
DB_PORT=27017
DB_NAME=your_database
DB_USER=your_username
DB_PASSWORD=your_password

# Or use a connection string
DB_URI=mongodb://user:password@host:port/database
```

### Generated API Structure

For each table in your database, the generator creates:

```
generated_api/
├── app.py                      # Main entry point
├── requirements.txt           # Dependencies
├── .env                       # Configuration
└── app/
    ├── main.py                # FastAPI application
    ├── core/
    │   └── config.py          # Settings management
    ├── models/                # Database models
    │   ├── database.py        # Database connection/engine
    │   ├── students.py        # SQLAlchemy ORM models (ORM mode)
    │   ├── courses.py
    │   └── ...
    ├── schemas/               # Pydantic validation schemas
    │   ├── students.py        # Request/response schemas
    │   ├── courses.py
    │   └── ...
    ├── services/              # Business logic layer
    │   ├── students_service.py
    │   ├── courses_service.py
    │   └── ...
    └── api/
        └── v1/
            └── endpoints/     # REST API endpoints
                ├── students.py
                ├── courses.py
                └── ...
```

### API Endpoints

Each table automatically gets these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/{table}/` | List all items (paginated) |
| `GET` | `/api/v1/{table}/{id}` | Get single item |
| `POST` | `/api/v1/{table}/` | Create new item |
| `PUT` | `/api/v1/{table}/{id}` | Full update |
| `PATCH` | `/api/v1/{table}/{id}` | Partial update |
| `DELETE` | `/api/v1/{table}/{id}` | Delete item |
| `GET` | `/api/v1/{table}/count/total` | Get total count |

### Query Parameters

All list endpoints support:

- `skip` - Number of items to skip (default: 0)
- `limit` - Maximum items to return (default: 100, max: 1000)
- `sort_by` - Column name to sort by
- `order` - Sort direction: `asc` or `desc` (default: `asc`)

### Example Usage

```bash
# Get all users with pagination
curl "http://localhost:8000/api/v1/users/?skip=0&limit=10&sort_by=created_at&order=desc"

# Get user by ID
curl "http://localhost:8000/api/v1/users/1"

# Create a new user
curl -X POST "http://localhost:8000/api/v1/users/" \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "email": "john@example.com"}'

# Update user
curl -X PUT "http://localhost:8000/api/v1/users/1" \
  -H "Content-Type: application/json" \
  -d '{"name": "Jane Doe", "email": "jane@example.com"}'

# Delete user
curl -X DELETE "http://localhost:8000/api/v1/users/1"
```

## 🏗️ Architecture

The generated projects follow industry best practices:

- **Separation of Concerns**: Models, Services, and Routes are cleanly separated
- **Service Layer**: Business logic lives in services, not in routes
- **Type Safety**: Full type hints with Pydantic models
- **Dependency Injection**: Ready for dependency injection patterns
- **Error Handling**: Proper HTTP status codes and error messages

## 🔧 How It Works

1. **Connect** to your database using the configured settings
2. **Discover** all tables/collections and their schemas automatically
3. **Generate** REST API endpoints for each table
4. **Create** Pydantic models dynamically based on database schema
5. **Implement** full CRUD operations with proper validation

## 🗄️ Database Support

### SQL Databases (MySQL, PostgreSQL, SQLite)

- Automatic primary key detection
- Foreign key relationship awareness
- Column type inference
- Transaction support
- SQLAlchemy ORM integration

### MongoDB

- Schema inference from sample documents
- Automatic ObjectId handling
- Collection-based operations
- Flexible document structure support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Database support via [SQLAlchemy](https://www.sqlalchemy.org/)
- Schema validation with [Pydantic](https://docs.pydantic.dev/)

## 📧 Support

For issues, questions, or contributions, please open an issue on [GitHub](https://github.com/yourusername/fastapi-schema-generator/issues).

---

Made with ❤️ for the FastAPI community
