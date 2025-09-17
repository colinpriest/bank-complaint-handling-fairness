#!/usr/bin/env python3
"""
Test the DatabaseCheck class
"""

from database_check import DatabaseCheck

def test_database_check():
    """Test basic DatabaseCheck functionality"""

    print("="*80)
    print("TESTING DATABASE_CHECK CLASS")
    print("="*80)

    # Initialize DatabaseCheck
    db_checker = DatabaseCheck()

    print(f"Database configuration:")
    print(f"  Host: {db_checker.db_host}")
    print(f"  Port: {db_checker.db_port}")
    print(f"  Database: {db_checker.db_name}")
    print(f"  User: {db_checker.db_user}")

    try:
        # Test PostgreSQL check
        if not db_checker.check_postgresql_installation():
            print("[ERROR] PostgreSQL check failed")
            return False

        # Test database connection
        if not db_checker.connect_to_database():
            print("[ERROR] Database connection failed")
            return False

        # Test table counts
        counts = db_checker.get_table_counts()
        print("\nTable counts:")
        for table, count in counts.items():
            print(f"  - {table}: {count:,}")

        # Test individual table checks
        print("\nTable existence checks:")
        print(f"  - Ground truth: {db_checker.check_ground_truth_table()}")
        print(f"  - LLM cache: {db_checker.check_llm_cache_table()}")
        print(f"  - Experiments: {db_checker.check_experiment_table()}")
        print(f"  - N-shot optimization: {db_checker.check_nshot_optimisation_table()}")

        print("\n[SUCCESS] DatabaseCheck class is working correctly!")
        return True

    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        db_checker.close_connection()

if __name__ == "__main__":
    test_database_check()