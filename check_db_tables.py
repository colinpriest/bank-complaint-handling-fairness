#!/usr/bin/env python3
import sqlite3

def check_database():
    conn = sqlite3.connect('complaints_analysis.db')
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print('Tables in database:')
    for table in tables:
        print(f'  {table[0]}')
    
    # Check if there are any tables with 'experiment' in the name
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%experiment%'")
    experiment_tables = cursor.fetchall()
    
    if experiment_tables:
        print()
        print('Tables containing "experiment":')
        for table in experiment_tables:
            table_name = table[0]
            print(f'  {table_name}')
            
            # Get count for each experiment table
            try:
                cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
                count = cursor.fetchone()[0]
                print(f'    Count: {count:,}')
                
                # Get some sample data structure
                cursor.execute(f'PRAGMA table_info({table_name})')
                columns = cursor.fetchall()
                print(f'    Columns: {[col[1] for col in columns[:5]]}...' if len(columns) > 5 else f'    Columns: {[col[1] for col in columns]}')
                
            except Exception as e:
                print(f'    Error getting info: {e}')
    else:
        print()
        print('No tables containing "experiment" found.')
    
    conn.close()

if __name__ == "__main__":
    check_database()
