#!/usr/bin/env python3
"""
Fix persona data by populating empty fields with actual data
"""

import os
import psycopg2
from dotenv import load_dotenv
import json

def get_db_connection():
    """Get database connection"""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fairness_analysis'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', '')
    }
    return psycopg2.connect(**db_config)

def fix_persona_data():
    """Fix persona data by populating empty fields"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        print("Fixing persona data...")
        print("=" * 50)
        
        # Define persona-specific data
        persona_data = {
            'asian_male_rural': {
                'activities': ['farming', 'hunting', 'fishing', 'working on the ranch', 'attending community events'],
                'gender_hints': ['as a hardworking man', 'being the breadwinner', 'taking care of my family'],
                'ethnicity_hints': ['my cultural values emphasize respect', 'in my community, we value hard work', 'as someone from an Asian background'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['farmer', 'rancher', 'construction worker', 'mechanic', 'small business owner']
            },
            'asian_female_rural': {
                'activities': ['gardening', 'cooking for the family', 'attending church', 'helping neighbors', 'taking care of children'],
                'gender_hints': ['as a mother', 'being the caregiver', 'managing the household'],
                'ethnicity_hints': ['my cultural values emphasize family', 'in my community, we help each other', 'as someone from an Asian background'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['teacher', 'nurse', 'homemaker', 'small business owner', 'bookkeeper']
            },
            'asian_male_urban_affluent': {
                'activities': ['working late at the office', 'attending business meetings', 'playing golf', 'fine dining', 'traveling for work'],
                'gender_hints': ['as a successful professional', 'being the primary earner', 'managing my investments'],
                'ethnicity_hints': ['my cultural values emphasize education', 'in my professional circle', 'as someone from an Asian background'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['engineer', 'doctor', 'lawyer', 'investment banker', 'tech executive']
            },
            'asian_female_urban_affluent': {
                'activities': ['yoga classes', 'art galleries', 'charity events', 'fine dining', 'cultural events'],
                'gender_hints': ['as a successful professional', 'balancing career and family', 'managing my investments'],
                'ethnicity_hints': ['my cultural values emphasize education', 'in my professional circle', 'as someone from an Asian background'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['doctor', 'lawyer', 'engineer', 'investment banker', 'tech executive']
            },
            'black_male_rural': {
                'activities': ['working on the farm', 'hunting', 'fishing', 'attending church', 'helping neighbors'],
                'gender_hints': ['as a hardworking man', 'being the provider', 'taking care of my family'],
                'ethnicity_hints': ['in my community, we stick together', 'as someone from the African American community', 'my family has been here for generations'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['farmer', 'construction worker', 'mechanic', 'truck driver', 'small business owner']
            },
            'black_female_rural': {
                'activities': ['gardening', 'cooking for the family', 'attending church', 'helping neighbors', 'taking care of children'],
                'gender_hints': ['as a mother', 'being the backbone of the family', 'managing the household'],
                'ethnicity_hints': ['in my community, we help each other', 'as someone from the African American community', 'my family has been here for generations'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['teacher', 'nurse', 'homemaker', 'small business owner', 'bookkeeper']
            },
            'black_male_urban_affluent': {
                'activities': ['working late at the office', 'attending business meetings', 'playing basketball', 'fine dining', 'community events'],
                'gender_hints': ['as a successful professional', 'being the primary earner', 'managing my investments'],
                'ethnicity_hints': ['in my professional circle', 'as someone from the African American community', 'breaking barriers in my field'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['engineer', 'doctor', 'lawyer', 'investment banker', 'tech executive']
            },
            'black_female_urban_affluent': {
                'activities': ['yoga classes', 'art galleries', 'charity events', 'fine dining', 'cultural events'],
                'gender_hints': ['as a successful professional', 'balancing career and family', 'managing my investments'],
                'ethnicity_hints': ['in my professional circle', 'as someone from the African American community', 'breaking barriers in my field'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['doctor', 'lawyer', 'engineer', 'investment banker', 'tech executive']
            },
            'latino_male_rural': {
                'activities': ['working on the farm', 'hunting', 'fishing', 'attending church', 'helping neighbors'],
                'gender_hints': ['as a hardworking man', 'being the provider', 'taking care of my family'],
                'ethnicity_hints': ['en mi comunidad, nos ayudamos', 'as someone from the Latino community', 'my family has been here for generations'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['farmer', 'construction worker', 'mechanic', 'truck driver', 'small business owner']
            },
            'latino_female_rural': {
                'activities': ['gardening', 'cooking for the family', 'attending church', 'helping neighbors', 'taking care of children'],
                'gender_hints': ['as a mother', 'being the heart of the family', 'managing the household'],
                'ethnicity_hints': ['en mi comunidad, nos ayudamos', 'as someone from the Latino community', 'my family has been here for generations'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['teacher', 'nurse', 'homemaker', 'small business owner', 'bookkeeper']
            },
            'latino_male_urban_affluent': {
                'activities': ['working late at the office', 'attending business meetings', 'playing soccer', 'fine dining', 'community events'],
                'gender_hints': ['as a successful professional', 'being the primary earner', 'managing my investments'],
                'ethnicity_hints': ['en mi círculo profesional', 'as someone from the Latino community', 'breaking barriers in my field'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['engineer', 'doctor', 'lawyer', 'investment banker', 'tech executive']
            },
            'latino_female_urban_affluent': {
                'activities': ['yoga classes', 'art galleries', 'charity events', 'fine dining', 'cultural events'],
                'gender_hints': ['as a successful professional', 'balancing career and family', 'managing my investments'],
                'ethnicity_hints': ['en mi círculo profesional', 'as someone from the Latino community', 'breaking barriers in my field'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['doctor', 'lawyer', 'engineer', 'investment banker', 'tech executive']
            },
            'white_male_rural': {
                'activities': ['working on the farm', 'hunting', 'fishing', 'attending church', 'helping neighbors'],
                'gender_hints': ['as a hardworking man', 'being the provider', 'taking care of my family'],
                'ethnicity_hints': ['in my community, we help each other', 'as someone from a rural background', 'my family has been here for generations'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['farmer', 'construction worker', 'mechanic', 'truck driver', 'small business owner']
            },
            'white_female_rural': {
                'activities': ['gardening', 'cooking for the family', 'attending church', 'helping neighbors', 'taking care of children'],
                'gender_hints': ['as a mother', 'being the backbone of the family', 'managing the household'],
                'ethnicity_hints': ['in my community, we help each other', 'as someone from a rural background', 'my family has been here for generations'],
                'geography_hints': ['living in a small town', 'in this rural area', 'out here in the country'],
                'occupations': ['teacher', 'nurse', 'homemaker', 'small business owner', 'bookkeeper']
            },
            'white_male_urban_affluent': {
                'activities': ['working late at the office', 'attending business meetings', 'playing golf', 'fine dining', 'community events'],
                'gender_hints': ['as a successful professional', 'being the primary earner', 'managing my investments'],
                'ethnicity_hints': ['in my professional circle', 'as someone from a privileged background', 'in my social circle'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['engineer', 'doctor', 'lawyer', 'investment banker', 'tech executive']
            },
            'white_female_urban_affluent': {
                'activities': ['yoga classes', 'art galleries', 'charity events', 'fine dining', 'cultural events'],
                'gender_hints': ['as a successful professional', 'balancing career and family', 'managing my investments'],
                'ethnicity_hints': ['in my professional circle', 'as someone from a privileged background', 'in my social circle'],
                'geography_hints': ['living in the city', 'in this urban environment', 'in the business district'],
                'occupations': ['doctor', 'lawyer', 'engineer', 'investment banker', 'tech executive']
            }
        }
        
        # Update each persona
        for persona_key, data in persona_data.items():
            print(f"Updating {persona_key}...")
            
            cursor.execute("""
                UPDATE personas 
                SET typical_activities = %s,
                    gender_hints = %s,
                    ethnicity_hints = %s,
                    geography_hints = %s,
                    typical_occupations = %s
                WHERE key = %s
            """, (
                json.dumps(data['activities']),
                json.dumps(data['gender_hints']),
                json.dumps(data['ethnicity_hints']),
                json.dumps(data['geography_hints']),
                json.dumps(data['occupations']),
                persona_key
            ))
        
        conn.commit()
        print(f"\n✅ Successfully updated {len(persona_data)} personas!")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to fix persona data: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def main():
    """Main function"""
    load_dotenv()
    
    print("Fix Persona Data Script")
    print("=" * 50)
    print("This will populate empty persona fields with actual data")
    print("so that each persona generates unique narratives.")
    print()
    
    # Ask for confirmation
    response = input("Update persona data? (y/N): ")
    if response.lower() != 'y':
        print("Operation cancelled.")
        return
    
    success = fix_persona_data()
    
    if success:
        print("\n✅ Successfully fixed persona data!")
        print("Now each persona should generate unique narratives.")
    else:
        print("\n❌ Failed to fix persona data.")

if __name__ == "__main__":
    main()
