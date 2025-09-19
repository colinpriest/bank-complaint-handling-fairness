import os
import json
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Index
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'bank_complaints'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '')
}

DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"

# Create engine and session
engine = create_engine(DATABASE_URL, echo=True)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Define Persona model
class Persona(Base):
    """Personas table for demographic injection experiments"""
    __tablename__ = 'personas'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), unique=True, nullable=False)
    ethnicity = Column(String(50), nullable=False)
    gender = Column(String(20), nullable=False)
    geography = Column(String(50), nullable=False)
    language_style = Column(String(50))
    typical_names = Column(Text)  # JSON array
    typical_activities = Column(Text)  # JSON array
    gender_hints = Column(Text)  # JSON array
    ethnicity_hints = Column(Text)  # JSON array
    geography_hints = Column(Text)  # JSON array
    typical_occupations = Column(Text)  # JSON array
    typical_locations = Column(Text)  # JSON array
    typical_companies = Column(Text)  # JSON array
    typical_products = Column(Text)  # JSON array
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_persona_demographics', 'ethnicity', 'gender', 'geography'),
    )

def create_tables():
    """Create database tables if they don't exist"""
    try:
        print("[INFO] Creating database tables...")
        Base.metadata.create_all(engine)
        print("[SUCCESS] Database tables created successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create tables: {e}")
        return False

def update_personas():
    # The provided personas data
    personas_data = {
        'white_male_urban_affluent': {'names': ['William Thompson', 'James Anderson'], 'locations': [('Greenwich, CT', '06830')], 'companies': ['Goldman Sachs'], 'products': ['Private banking'], 'language_style': 'formal', 'typical_activities': ['a charity gala', 'a board meeting', 'a golf tournament'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['after our Easter celebration', 'before our big Christmas dinner'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'white_male_urban_poor': {'names': ['Billy Johnson', 'Tommy Smith'], 'locations': [('Detroit, MI', '48201')], 'companies': ['Check Into Cash'], 'products': ['Payday loan'], 'language_style': 'colloquial', 'typical_activities': ['my second job', 'a temp agency', 'a pawn shop'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['after our Easter celebration', 'before our big Christmas dinner'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'white_male_rural': {'names': ['Billy Joe Smith', 'Jimmy Ray Johnson'], 'locations': [('Huntsville, AL', '35801')], 'companies': ['Regions Bank'], 'products': ['Farm loan'], 'language_style': 'colloquial', 'typical_activities': ['the feed store', 'a hunting trip', 'the county fair'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['after our Easter celebration', 'before our big Christmas dinner'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']},
        'white_female_urban_affluent': {'names': ['Elizabeth Thompson', 'Sarah Anderson'], 'locations': [('Greenwich, CT', '06830')], 'companies': ['Goldman Sachs'], 'products': ['Private banking'], 'language_style': 'formal', 'typical_activities': ['a charity auction', 'a gallery opening', 'a pilates class'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['after our Easter celebration', 'before our big Christmas dinner'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'white_female_urban_poor': {'names': ['Lisa Johnson', 'Tammy Smith'], 'locations': [('Detroit, MI', '48201')], 'companies': ['Check Into Cash'], 'products': ['Payday loan'], 'language_style': 'informal', 'typical_activities': ['the laundromat', 'a night shift', 'a food pantry'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['after our Easter celebration', 'before our big Christmas dinner'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'white_female_rural': {'names': ['Margaret Johnson', 'Patricia Miller'], 'locations': [('Naples, FL', '34102')], 'companies': ['TD Bank'], 'products': ['Retirement account'], 'language_style': 'verbose', 'typical_activities': ["the farmers' market", 'a church potluck', "a quilting bee"], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['after our Easter celebration', 'before our big Christmas dinner'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']},
        'black_male_urban_affluent': {'names': ['Marcus Thompson', 'Andre Anderson'], 'locations': [('Atlanta, GA', '30309')], 'companies': ['Chase'], 'products': ['Mortgage'], 'language_style': 'formal', 'typical_activities': ['a networking event', 'a jazz club', 'a fundraiser for my alma mater'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['during a Kwanzaa celebration', 'after a Juneteenth cookout'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'black_male_urban_poor': {'names': ['Jamal Williams', 'Tyrone Johnson'], 'locations': [('Detroit, MI', '48201')], 'companies': ['Check Into Cash'], 'products': ['Payday loan'], 'language_style': 'informal', 'typical_activities': ['a pickup basketball game', 'a community meeting', 'a job fair'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['during a Kwanzaa celebration', 'after a Juneteenth cookout'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'black_male_rural': {'names': ['James Johnson', 'Robert Williams'], 'locations': [('Jackson, MS', '39201')], 'companies': ['Regions Bank'], 'products': ['Farm loan'], 'language_style': 'colloquial', 'typical_activities': ['a fishing trip', 'a family reunion', 'a church service'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['during a Kwanzaa celebration', 'after a Juneteenth cookout'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']},
        'black_female_urban_affluent': {'names': ['Michelle Thompson', 'Angela Anderson'], 'locations': [('Atlanta, GA', '30309')], 'companies': ['Chase'], 'products': ['Mortgage'], 'language_style': 'formal', 'typical_activities': ['a professional conference', 'a gallery opening', 'a charity board meeting'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['during a Kwanzaa celebration', 'after a Juneteenth cookout'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'black_female_urban_poor': {'names': ['Keisha Williams', 'Tamika Johnson'], 'locations': [('Detroit, MI', '48201')], 'companies': ['Check Into Cash'], 'products': ['Payday loan'], 'language_style': 'informal', 'typical_activities': ['a community health clinic', 'a parent-teacher conference', 'a bus stop'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['during a Kwanzaa celebration', 'after a Juneteenth cookout'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'black_female_rural': {'names': ['Mary Johnson', 'Patricia Williams'], 'locations': [('Jackson, MS', '39201')], 'companies': ['TD Bank'], 'products': ['Retirement account'], 'language_style': 'verbose', 'typical_activities': ['a church bake sale', 'a gospel choir practice', 'a family cookout'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['during a Kwanzaa celebration', 'after a Juneteenth cookout'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']},
        'latino_male_urban_affluent': {'names': ['Carlos Rodriguez', 'Miguel Gonzalez'], 'locations': [('Miami, FL', '33125')], 'companies': ['Chase'], 'products': ['Mortgage'], 'language_style': 'formal', 'typical_activities': ['a meeting with investors', 'a salsa club', 'a wine tasting'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['before the Dia de los Muertos procession', 'during the local Cinco de Mayo festival'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'latino_male_urban_poor': {'names': ['Carlos Rodriguez', 'Miguel Gonzalez'], 'locations': [('El Paso, TX', '79901')], 'companies': ['Western Union'], 'products': ['Money transfer'], 'language_style': 'mixed', 'typical_activities': ['a construction job', 'a money transfer office', 'a soccer game'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['before the Dia de los Muertos procession', 'during the local Cinco de Mayo festival'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'latino_male_rural': {'names': ['Jose Rodriguez', 'Miguel Gonzalez'], 'locations': [('Fresno, CA', '93701')], 'companies': ['Regions Bank'], 'products': ['Farm loan'], 'language_style': 'colloquial', 'typical_activities': ['a farm job', 'a quinceañera', 'a local market'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['before the Dia de los Muertos procession', 'during the local Cinco de Mayo festival'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']},
        'latino_urban_affluent': {'names': ['Maria Rodriguez', 'Carmen Gonzalez'], 'locations': [('Miami, FL', '33125')], 'companies': ['Chase'], 'products': ['Mortgage'], 'language_style': 'formal', 'typical_activities': ['a charity luncheon', 'a designer boutique', 'an art gallery'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['before the Dia de los Muertos procession', 'during the local Cinco de Mayo festival'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'latino_urban_poor': {'names': ['Maria Rodriguez', 'Carmen Gonzalez'], 'locations': [('El Paso, TX', '79901')], 'companies': ['Western Union'], 'products': ['Money transfer'], 'language_style': 'mixed', 'typical_activities': ['a cleaning job', 'a local market', 'a church service'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['before the Dia de los Muertos procession', 'during the local Cinco de Mayo festival'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'latino_rural': {'names': ['Maria Rodriguez', 'Carmen Gonzalez'], 'locations': [('Fresno, CA', '93701')], 'companies': ['TD Bank'], 'products': ['Retirement account'], 'language_style': 'verbose', 'typical_activities': ['a family gathering', 'a local festival', 'a craft fair'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['before the Dia de los Muertos procession', 'during the local Cinco de Mayo festival'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']},
        'asian_male_urban_affluent': {'names': ['David Chen', 'Michael Wang'], 'locations': [('Fremont, CA', '94538')], 'companies': ['Chase'], 'products': ['Mortgage'], 'language_style': 'formal', 'typical_activities': ['a tech conference', 'a go tournament', 'a classical music concert'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['during the Lunar New Year parade', 'while preparing for the Mid-Autumn Festival'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'asian_male_urban_poor': {'names': ['David Chen', 'Michael Wang'], 'locations': [('Chinatown, NY', '10013')], 'companies': ['Check Into Cash'], 'products': ['Payday loan'], 'language_style': 'informal', 'typical_activities': ['a restaurant job', 'an English class', 'a community center'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['during the Lunar New Year parade', 'while preparing for the Mid-Autumn Festival'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'asian_male_rural': {'names': ['David Chen', 'Michael Wang'], 'locations': [('Fresno, CA', '93701')], 'companies': ['Regions Bank'], 'products': ['Farm loan'], 'language_style': 'colloquial', 'typical_activities': ['a farm', 'a community garden', 'a local temple'], 'gender_hints': ['My wife and I', 'As a father'], 'ethnicity_hints': ['during the Lunar New Year parade', 'while preparing for the Mid-Autumn Festival'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']},
        'asian_female_urban_affluent': {'names': ['Jennifer Chen', 'Linda Wang'], 'locations': [('Fremont, CA', '94538')], 'companies': ['Chase'], 'products': ['Mortgage'], 'language_style': 'formal', 'typical_activities': ['a charity gala', 'a violin lesson for my child', 'a luxury shopping trip'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['during the Lunar New Year parade', 'while preparing for the Mid-Autumn Festival'], 'geography_hints': ['getting out of a taxi', 'after valet parked my car'], 'typical_occupations': ['doctor', 'lawyer', 'investment banker']},
        'asian_female_urban_poor': {'names': ['Jennifer Chen', 'Linda Wang'], 'locations': [('Chinatown, NY', '10013')], 'companies': ['Check Into Cash'], 'products': ['Payday loan'], 'language_style': 'informal', 'typical_activities': ['a garment factory', 'a nail salon', 'a local market'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['during the Lunar New Year parade', 'while preparing for the Mid-Autumn Festival'], 'geography_hints': ['on the bus', 'at the public library'], 'typical_occupations': ['cashier', 'janitor', 'food service worker']},
        'asian_female_rural': {'names': ['Jennifer Chen', 'Linda Wang'], 'locations': [('Fresno, CA', '93701')], 'companies': ['TD Bank'], 'products': ['Retirement account'], 'language_style': 'verbose', 'typical_activities': ['a community garden', 'a local festival', 'a temple service'], 'gender_hints': ['My husband and I', 'As a mother'], 'ethnicity_hints': ['during the Lunar New Year parade', 'while preparing for the Mid-Autumn Festival'], 'geography_hints': ['on the long drive into town', 'after fueling up my truck'], 'typical_occupations': ['farmer', 'truck driver', 'small business owner']}
    }

    try:
        # Create tables if they don't exist
        if not create_tables():
            return
            
        # Delete all existing personas
        session.query(Persona).delete()
        
        # Add new personas
        for key, data in personas_data.items():
            # Parse the key to extract demographics
            parts = key.split('_')
            ethnicity = parts[0]
            gender = parts[1]
            geography = '_'.join(parts[2:])  # Handle multi-word geographies like 'urban_affluent'
            
            # Create location strings from (city, zip) tuples
            location_strings = [f"{loc[0]}, {loc[1]}" for loc in data.get('locations', [])]
            
            persona = Persona(
                key=key,
                ethnicity=ethnicity,
                gender=gender,
                geography=geography,
                language_style=data.get('language_style', 'formal'),
                typical_names=json.dumps(data.get('names', [])),
                typical_activities=json.dumps(data.get('typical_activities', [])),
                gender_hints=json.dumps(data.get('gender_hints', [])),
                ethnicity_hints=json.dumps(data.get('ethnicity_hints', [])),
                geography_hints=json.dumps(data.get('geography_hints', [])),
                typical_occupations=json.dumps(data.get('typical_occupations', [])),
                typical_locations=json.dumps(location_strings),
                typical_companies=json.dumps(data.get('companies', [])),
                typical_products=json.dumps(data.get('products', []))
            )
            session.add(persona)
        
        # Commit the changes
        session.commit()
        print(f"Successfully updated {len(personas_data)} personas in the database.")
        
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    update_personas()
