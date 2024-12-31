from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import mysql.connector
from mysql.connector import Error as MySQLError, pooling
from contextlib import contextmanager
import logging
import os
from config import Config

class DatabaseManager:
    def __init__(self):
        self.mongo_client = None
        self.mysql_pool = None
        self.logger = logging.getLogger(__name__)
        self._init_connections()

    def _init_connections(self):
        """Initialize database connections"""
        try:
            # Initialize MongoDB
            self.mongo_client = MongoClient(
                Config.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            # Test MongoDB connection
            self.mongo_client.admin.command('ping')
            self.logger.info("MongoDB connection successful")
            
            # Initialize MongoDB collections
            db = self.mongo_client[Config.MONGODB_DB]
            collections = ['uploads', 'model_reports']
            for collection in collections:
                if collection not in db.list_collection_names():
                    db.create_collection(collection)

            # Initialize MySQL connection pool
            dbconfig = {
                'user': Config.MYSQL_CONFIG['user'],
                'password': Config.MYSQL_CONFIG['password'],
                'host': Config.MYSQL_CONFIG['host'],
                'database': Config.MYSQL_CONFIG['database'],
                'raise_on_warnings': True,
                'auth_plugin': 'mysql_native_password',
                'pool_name': 'mypool',
                'pool_size': 5
            }
            self.mysql_pool = mysql.connector.pooling.MySQLConnectionPool(**dbconfig)
            self.logger.info("MySQL connection pool initialized")

        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            if self.mongo_client:
                self.mongo_client.close()
            if self.mysql_pool:
                self.mysql_pool = None
            raise

    @contextmanager
    def get_mongo_connection(self):
        """Get MongoDB connection using context manager."""
        try:
            if not self.mongo_client:
                self._init_connections()
            
            db = self.mongo_client[Config.MONGODB_DB]
            yield db
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"MongoDB connection error: {e}")
            self.mongo_client = None
            raise ConnectionError(f"MongoDB connection failed: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"MongoDB error: {e}")
            raise

    @contextmanager
    def get_mysql_connection(self):
        """Get MySQL connection using context manager."""
        conn = None
        try:
            conn = self.mysql_pool.get_connection()
            yield conn
            conn.commit()
            
        except mysql.connector.Error as e:
            self.logger.error(f"MySQL error: {e}")
            if conn:
                conn.rollback()
            raise
            
        finally:
            if conn and conn.is_connected():
                conn.close()

    def initialize_database(self):
        """Initialize the database and create tables if they don't exist."""
        try:
            with self.get_mysql_connection() as conn:
                cursor = conn.cursor()
                
                # Check if users table exists
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_name = 'users'
                """, (Config.MYSQL_CONFIG['database'],))
                
                table_exists = cursor.fetchone()[0] > 0
                
                if not table_exists:
                    # Create users table
                    cursor.execute("""
                        CREATE TABLE users (
                            id INT AUTO_INCREMENT PRIMARY KEY,
                            username VARCHAR(50) UNIQUE NOT NULL,
                            password VARCHAR(255) NOT NULL,
                            email VARCHAR(120) UNIQUE NOT NULL,
                            name VARCHAR(100),
                            organization VARCHAR(100),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            last_login TIMESTAMP NULL,
                            active BOOLEAN DEFAULT TRUE
                        )
                    """)
                    self.logger.info("Users table created successfully")
                else:
                    self.logger.info("Users table already exists")

                cursor.close()
                return True

        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            return False

    def test_connections(self):
        """Test all database connections."""
        try:
            # Test MongoDB
            with self.get_mongo_connection() as mongo_db:
                mongo_db.command('ping')
                self.logger.info("MongoDB connection test successful")

            # Test MySQL
            with self.get_mysql_connection() as mysql_conn:
                cursor = mysql_conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                self.logger.info("MySQL connection test successful")

            return True

        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False

    def close_connections(self):
        """Close all database connections."""
        try:
            if self.mongo_client:
                self.mongo_client.close()
                self.mongo_client = None
            if self.mysql_pool:
                # Close all connections in the pool
                for _ in range(self.mysql_pool._cnx_queue.qsize()):
                    try:
                        conn = self.mysql_pool.get_connection()
                        conn.close()
                    except:
                        pass
            self.logger.info("All database connections closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize database manager
db_manager = DatabaseManager()

# Initialize database
if not db_manager.initialize_database():
    logging.warning("Database initialization incomplete - some features may not work")