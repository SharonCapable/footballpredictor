import psycopg2

try:
    # Try to connect
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="football_predictions",
        user="football_user",
        password="FootballPred2025!"
    )
    
    print("✓ PostgreSQL connection successful!")
    
    # Test a simple query
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"✓ PostgreSQL version: {version[0]}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"✗ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check PostgreSQL is running")
    print("2. Verify password is correct")
    print("3. Ensure database 'football_predictions' exists")