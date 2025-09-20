#!/usr/bin/env python3
import os
import psycopg2
from dotenv import load_dotenv

def check_postgres_experiments():
    # Load environment variables
    load_dotenv()
    
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    
    print("Database Configuration:")
    print(f"  Host: {db_config['host']}")
    print(f"  Port: {db_config['port']}")
    print(f"  Database: {db_config['database']}")
    print(f"  User: {db_config['user']}")
    print()
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Check if experiments table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'experiments'
            );
        """)
        
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("❌ 'experiments' table does not exist in the database.")
            print("\nAvailable tables:")
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            for table in tables:
                print(f"  - {table[0]}")
            return
        
        # Get total experiment count
        cursor.execute('SELECT COUNT(*) FROM experiments')
        total_experiments = cursor.fetchone()[0]
        
        # Get experiments by decision method
        cursor.execute('SELECT decision_method, COUNT(*) FROM experiments GROUP BY decision_method ORDER BY decision_method')
        method_counts = cursor.fetchall()
        
        # Get experiments by geography
        cursor.execute('''
            SELECT geography, COUNT(*) 
            FROM experiments 
            WHERE geography IS NOT NULL 
            GROUP BY geography 
            ORDER BY COUNT(*) DESC
        ''')
        geography_counts = cursor.fetchall()
        
        # Get experiments by completion status
        cursor.execute('SELECT COUNT(*) FROM experiments WHERE llm_simplified_tier = -999')
        pending_experiments = cursor.fetchone()[0]
        completed_experiments = total_experiments - pending_experiments
        
        # Get experiments by persona
        cursor.execute('''
            SELECT persona, COUNT(*) 
            FROM experiments 
            WHERE persona IS NOT NULL 
            GROUP BY persona 
            ORDER BY COUNT(*) DESC
            LIMIT 10
        ''')
        persona_counts = cursor.fetchall()
        
        print("📊 EXPERIMENT STATISTICS")
        print("=" * 50)
        print(f"Total Experiments: {total_experiments:,}")
        print(f"Completed: {completed_experiments:,}")
        print(f"Pending (need LLM analysis): {pending_experiments:,}")
        if total_experiments > 0:
            print(f"Completion Rate: {(completed_experiments/total_experiments*100):.1f}%")
        print()
        
        print("📋 BY DECISION METHOD:")
        for method, count in method_counts:
            print(f"  {method}: {count:,}")
        print()
        
        print("🌍 BY GEOGRAPHY:")
        for geo, count in geography_counts:
            print(f"  {geo}: {count:,}")
        print()
        
        print("👤 TOP 10 PERSONAS:")
        for persona, count in persona_counts:
            print(f"  {persona}: {count:,}")
        print()
        
        # Check for new geographic categories
        new_geographies = ['urban_upper_middle', 'urban_working', 'suburban_upper_middle', 
                          'suburban_working', 'suburban_poor', 'rural_upper_middle', 
                          'rural_working', 'rural_poor']
        
        existing_geographies = [geo[0] for geo in geography_counts]
        missing_geographies = [geo for geo in new_geographies if geo not in existing_geographies]
        
        if missing_geographies:
            print("⚠️  MISSING NEW GEOGRAPHIC CATEGORIES:")
            for geo in missing_geographies:
                print(f"  - {geo}")
            print()
            print("💡 To add these categories, run:")
            print("   python add_geographic_persona_options.py")
        else:
            print("✅ All new geographic categories are present!")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has correct database credentials")
        print("2. Ensure PostgreSQL is running")
        print("3. Verify the database exists")

if __name__ == "__main__":
    check_postgres_experiments()
