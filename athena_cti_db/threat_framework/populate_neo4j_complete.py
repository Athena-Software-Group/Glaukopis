import csv
import decimal
import gzip
import io
import ijson
import json
import os
import shutil
import requests
import xml.etree.ElementTree as ET
import zipfile
import tempfile
import uuid
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime, timedelta
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import subprocess
import re
import logging
import base64
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get Neo4j configuration from environment variables
neo4j_url = os.getenv('NEO4J_URL', 'neo4j://127.0.0.1:7687')
neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
neo4j_password = os.getenv('NEO4J_PASSWORD', 'athena-cti-db')
neo4j_db = os.getenv('NEO4J_DB', 'athena-cti-db') # default database is neo4j, changes are required for other databases i.e., athena-threat-db

# NVD API key — with key: 50 req/30s; without: 5 req/30s
nvd_api_key = os.getenv('NVD_API_KEY', 'a3a68e0f-b241-42e3-927a-e8f1828f2910') 

# Check if using Bolt protocol (neo4j:// or bolt://)
use_bolt = neo4j_url.startswith('neo4j://') or neo4j_url.startswith('bolt://')

if use_bolt:
    from neo4j import GraphDatabase
    logger.info(f"Neo4j populate module using Bolt protocol: {neo4j_url}")
    logger.info(f"Using Neo4j user: {neo4j_user}, database: {neo4j_db}")
else:
    NEO4J_ENDPOINT = f"{neo4j_url}/db/neo4j/tx/commit"
    # Create base64 encoded auth header
    auth_string = f"{neo4j_user}:{neo4j_password}"
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii')

    HEADERS = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Basic {auth_b64}"
    }
    
    # Log the endpoint being used
    logger.info(f"Neo4j populate module using HTTP endpoint: {NEO4J_ENDPOINT}")
    logger.info(f"Using Neo4j user: {neo4j_user}")

# Bolt driver singleton — created once, reused across all execute_queries calls
_bolt_driver = None

def _get_bolt_driver():
    """Return a cached Bolt driver, creating it once on first call."""
    global _bolt_driver
    if _bolt_driver is None:
        _bolt_driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    return _bolt_driver

# Data source URLs
DATA_SOURCES = {
    "attack": "https://github.com/mitre/cti.git",
    "capec": "http://capec.mitre.org/data/xml/capec_latest.xml",
    "cwe": "http://cwe.mitre.org/data/xml/cwec_latest.xml.zip",
    "cve": "https://github.com/CVEProject/cvelistV5.git",
    "kev": "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
    "engage": "https://github.com/mitre/engage.git",
    "epss": "https://epss.cyentia.com/epss_scores-{date}.csv.gz",
    "nvd": "https://nvd.nist.gov/feeds/json/cve/2.0",
    "sigma": "https://github.com/SigmaHQ/sigma.git",
    "exploitdb": "https://gitlab.com/exploit-database/exploitdb.git",
    "poc_github": "https://github.com/nomi-sec/PoC-in-GitHub.git"
}

# Configure retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)

session = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

def generate_stix_id(framework: str, entity_id: str) -> str:
    """Generate a STIX-like ID for entities that don't have one."""
    namespace = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{framework}.mitre.org"))
    entity_uuid = str(uuid.uuid5(uuid.UUID(namespace), entity_id))
    return f"{framework}--{entity_uuid}"

def should_ignore_cve_file(file_path: Path) -> bool:
    """Check if a CVE file should be ignored during processing."""
    # Files to ignore in CVE processing
    ignored_files = {
        "cves/delta.json",
        "cves/deltaLog.json"
    }
    
    # Convert file path to relative path from cve directory for comparison
    try:
        # Get the relative path from the cve directory
        relative_path = file_path.parts[-2:] if len(file_path.parts) >= 2 else file_path.parts
        relative_path_str = "/".join(relative_path)
        return relative_path_str in ignored_files
    except Exception:
        return False

def should_ignore_cti_file(file_path: Path, dir_name: str) -> bool:
    """Check if a CTI file should be ignored during processing."""
    # Files to ignore in CTI processing - these are the consolidated files
    ignored_files = {
        "enterprise-attack.json",
        "mobile-attack.json", 
        "ics-attack.json",
        "pre-attack.json"
    }
    
    # Check if this is a consolidated file in the main directory
    if file_path.name in ignored_files and file_path.parent.name == dir_name:
        return True
    
    return False

def filter_cve_files(json_files: List[Path]) -> List[Path]:
    """Filter out ignored CVE files and files from years before 2024."""
    result = []
    for f in json_files:
        if should_ignore_cve_file(f):
            continue
        # Only include files inside a year folder >= 2024
        # CVE zip structure: cves/{year}/{range}/{CVE-YYYY-NNNNN}.json
        year_part = next(
            (part for part in f.parts if part.isdigit() and len(part) == 4),
            None
        )
        if year_part and int(year_part) >= 2024:
            result.append(f)
    return result

def filter_cti_files(json_files: List[Path], dir_name: str) -> List[Path]:
    """Filter out ignored CTI files from the list."""
    return [f for f in json_files if not should_ignore_cti_file(f, dir_name)]

def is_zip_file(file_path: Path) -> bool:
    """Check if a file is a zip file by examining its extension and magic bytes."""
    try:
        # Check file extension
        if file_path.suffix.lower() == '.zip':
            return True
        
        # Check magic bytes (zip files start with PK\x03\x04)
        with open(file_path, 'rb') as f:
            magic = f.read(4)
            return magic == b'PK\x03\x04'
    except Exception:
        return False

def extract_zip_recursively(zip_path: Path, extract_dir: Path) -> bool:
    """Extract zip file and recursively extract any nested zip files."""
    try:
        logger.info(f"Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Check if any extracted files still have .zip extension
        for extracted_file in extract_dir.rglob("*.zip"):
            if extracted_file.is_file():
                logger.info(f"Found nested zip: {extracted_file.name}")
                # Recursively extract the nested zip
                if extract_zip_recursively(extracted_file, extract_dir):
                    # Remove the nested zip file after successful extraction
                    try:
                        extracted_file.unlink()
                        logger.debug(f"Removed nested zip: {extracted_file.name}")
                    except OSError as e:
                        logger.warning(f"Could not remove nested zip {extracted_file.name}: {e}")
        
        return True
        
    except zipfile.BadZipFile as e:
        logger.error(f"Bad zip file {zip_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error extracting {zip_path.name}: {e}")
        return False

def download_and_extract_data():
    """Download and extract data from all sources."""
    data_dir = Path("threat_data")
    data_dir.mkdir(exist_ok=True)
    
    # Download ATT&CK data (git clone)
    attack_dir = data_dir / "cti"
    # Check if ATT&CK data already exists (look for enterprise-attack directory)
    attack_data_exists = (attack_dir / "enterprise-attack").exists()
    if not attack_data_exists:
        logger.info("Cloning MITRE CTI repository...")
        subprocess.run(["git", "clone", DATA_SOURCES["attack"], str(attack_dir)], check=True)
    else:
        logger.info("ATT&CK data already exists - skipping download")
    
    # Download CAPEC XML
    capec_file = data_dir / "capec_latest.xml"
    if not capec_file.exists():
        logger.info("Downloading CAPEC data...")
        with open(capec_file, 'wb') as f:
            with requests.get(DATA_SOURCES["capec"], stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    else:
        logger.info("CAPEC data already exists - skipping download")
    
    # Download and extract CWE XML
    cwe_dir = data_dir / "cwe"
    cwe_dir.mkdir(exist_ok=True)
    # Check if any XML file exists in CWE directory
    cwe_file_exists = any(cwe_dir.glob("*.xml"))
    if not cwe_file_exists:
        logger.info("Downloading and extracting CWE data...")
        
        # Stream download for consistency
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            temp_path = temp_file.name
            
            with requests.get(DATA_SOURCES["cwe"], stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
        
        # Use recursive extraction
        if extract_zip_recursively(Path(temp_path), cwe_dir):
            logger.info("CWE data downloaded and extracted successfully")
        else:
            logger.error("Failed to extract CWE data")
            raise Exception("CWE extraction failed")
        
        os.unlink(temp_path)
    else:
        logger.info("CWE data already exists - skipping download")
    
    # Download CVE data via sparse git clone (2024 onwards only)
    cve_dir = data_dir / "cve"
    cve_dir.mkdir(exist_ok=True)
    cve_files_exist = any(cve_dir.rglob("*.json"))
    if not cve_files_exist:
        logger.info("Cloning CVE repository with sparse checkout (2024 onwards)...")

        current_year = datetime.now().year
        sparse_paths = [f"cves/{year}" for year in range(2024, current_year + 1)]

        subprocess.run(
            ["git", "clone", "--no-checkout", "--depth=1", "--filter=blob:none",
             DATA_SOURCES["cve"], str(cve_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cve_dir), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cve_dir), "sparse-checkout", "set"] + sparse_paths,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cve_dir), "checkout"],
            check=True,
        )

        logger.info(f"CVE data cloned successfully. Years fetched: {', '.join(sparse_paths)}")
    else:
        logger.info("CVE data already exists - skipping download")
    
    # Download KEV data
    kev_file = data_dir / "known_exploited_vulnerabilities.json"
    if not kev_file.exists():
        logger.info("Downloading KEV data...")
        with open(kev_file, 'wb') as f:
            with requests.get(DATA_SOURCES["kev"], stream=True) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    else:
        logger.info("KEV data already exists - skipping download")

    # Download Engage data (git clone)
    engage_dir = data_dir / "engage"
    engage_data_exists = (engage_dir / "Data" / "json").exists()
    if not engage_data_exists:
        logger.info("Cloning MITRE Engage repository...")
        subprocess.run(["git", "clone", DATA_SOURCES["engage"], str(engage_dir)], check=True)
    else:
        logger.info("Engage data already exists - skipping download")

    # Download NVD CPE data (one file per year, 2024 onwards)
    download_nvd_data(data_dir)

    # Clone Sigma rules repository
    sigma_dir = data_dir / "sigma"
    sigma_data_exists = (sigma_dir / "rules").exists()
    if not sigma_data_exists:
        logger.info("Cloning SigmaHQ sigma repository (sparse: rules/ only)...")
        subprocess.run(
            ["git", "clone", "--no-checkout", "--depth=1", "--filter=blob:none",
             DATA_SOURCES["sigma"], str(sigma_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(sigma_dir), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(sigma_dir), "sparse-checkout", "set", "rules"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(sigma_dir), "checkout"],
            check=True,
        )
        logger.info("Sigma rules cloned successfully")
    else:
        logger.info("Sigma data already exists - skipping download")

    # Clone ExploitDB (sparse: files_exploits.csv only)
    exploitdb_dir = data_dir / "exploitdb"
    exploitdb_csv = exploitdb_dir / "files_exploits.csv"
    if not exploitdb_csv.exists():
        logger.info("Cloning ExploitDB repository (sparse: files_exploits.csv only)...")
        exploitdb_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", "--depth=1", "--filter=blob:none",
             DATA_SOURCES["exploitdb"], str(exploitdb_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(exploitdb_dir), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(exploitdb_dir), "sparse-checkout", "set", "files_exploits.csv"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(exploitdb_dir), "checkout"],
            check=True,
        )
        logger.info("ExploitDB data cloned successfully")
    else:
        logger.info("ExploitDB data already exists - skipping download")

    # Clone poc-in-github (sparse: 2024+ year folders)
    poc_dir = data_dir / "poc_github"
    poc_data_exists = any(poc_dir.glob("202*"))
    if not poc_data_exists:
        logger.info("Cloning poc-in-github repository (sparse: 2024 onwards)...")
        current_year = datetime.now().year
        poc_years = [str(year) for year in range(2024, current_year + 1)]
        poc_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", "--depth=1", "--filter=blob:none",
             DATA_SOURCES["poc_github"], str(poc_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(poc_dir), "sparse-checkout", "init", "--cone"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(poc_dir), "sparse-checkout", "set"] + poc_years,
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(poc_dir), "checkout"],
            check=True,
        )
        logger.info(f"poc-in-github data cloned successfully. Years fetched: {', '.join(poc_years)}")
    else:
        logger.info("poc-in-github data already exists - skipping download")

    return data_dir

def load_json_file(file_path: str) -> Dict:
    """Load and parse a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def to_neo4j_datetime(date_str):
    """Convert any ISO date/datetime string to a Neo4j datetime()-compatible format.

    Handles all formats found across data sources:
      - YYYY-MM-DD              (CWE, CAPEC, KEV)  → appends T00:00:00
      - YYYY-MM-DD HH:MM:SS     (space separator)  → replaces space with T
      - YYYY-MM-DDTHH:MM:SS     (ATT&CK, CVE)      → unchanged
      - YYYY-MM-DDTHH:MM:SS.sssZ (ATT&CK with ms)  → unchanged
    Returns 'N/A' for empty, null, or unparseable values.
    """
    if not date_str or date_str == "N/A":
        return "N/A"
    try:
        s = re.sub(r'\s*:\s*', ':', str(date_str).strip()).replace(' ', 'T')
        # Date-only string (YYYY-MM-DD): append time so Neo4j datetime() accepts it
        if len(s) == 10 and s[4] == '-' and s[7] == '-':
            s += 'T00:00:00'
        return s
    except:
        return "N/A"

def to_json_string(obj):
    """Convert dict/list to JSON string or return safe default."""
    if obj is None:
        return "none"
    if isinstance(obj, (dict, list)):
        return json.dumps(obj)
    return str(obj)

def create_node_query(node_type: str, stix_id: str, properties: Dict) -> str:
    """Create a Cypher query to create a node if it doesn't exist using MERGE."""
    all_properties = {**properties, 'stix_id': stix_id}
    # Datetime fields that need datetime() wrapper
    datetime_fields = {'created', 'modified', 'first_seen', 'last_seen', 'dateAdded', 'dueDate'}
    
    # Build property assignments with datetime() for date fields
    props_list = []
    for k in all_properties.keys():
        if k in datetime_fields and all_properties[k] != "N/A":
            props_list.append(f"{k}: datetime(${k})")
        else:
            props_list.append(f"{k}: ${k}")
    
    props_str = ", ".join(props_list)
    # Use backticks for labels with hyphens
    label = f"`{node_type}`" if '-' in node_type else node_type
    return f"MERGE (n:{label} {{stix_id: $stix_id}}) SET n += {{{props_str}}}"

def process_attack_data(cti_dir: Path) -> List[Dict]:
    """Process ATT&CK data from CTI repository."""
    queries = []
    enterprise_dir = cti_dir / "enterprise-attack"

    # Process Techniques
    all_technique_files = list((enterprise_dir / "attack-pattern").glob("*.json"))
    filtered_technique_files = filter_cti_files(all_technique_files, "attack-pattern")
    
    for file_path in filtered_technique_files:
        data = load_json_file(str(file_path))
        technique = data['objects'][0]
        
        mitre_id = next((ref['external_id'] for ref in technique.get('external_references', [])
                        if ref.get('source_name') == 'mitre-attack'), None)
        
        properties = {
            'created': to_neo4j_datetime(technique.get('created')),
            'description': technique.get('description', 'N/A'),
            'external_references': to_json_string(technique.get('external_references', [])),
            'id': technique.get('id', 'N/A'),
            'kill_chain_phases': to_json_string(technique.get('kill_chain_phases', [])),
            'mitre_id': mitre_id or 'N/A',
            'modified': to_neo4j_datetime(technique.get('modified')),
            'name': technique.get('name', 'N/A'),
            'type': technique.get('type', 'N/A'),
            'x_mitre_data_sources': technique.get('x_mitre_data_sources', []),
            'x_mitre_deprecated': technique.get('x_mitre_deprecated', False),
            'x_mitre_detection': technique.get('x_mitre_detection', 'N/A'),
            'x_mitre_domains': technique.get('x_mitre_domains', []),
            'x_mitre_is_subtechnique': technique.get('x_mitre_is_subtechnique', False),
            'x_mitre_platforms': technique.get('x_mitre_platforms', []),
            'x_mitre_version': technique.get('x_mitre_version', 'N/A'),
            'x_mitre_attack_spec_version': technique.get('x_mitre_attack_spec_version', 'N/A')
        }
        
        queries.append({
            'statement': create_node_query('attack-pattern', technique['id'], properties),
            'parameters': {**properties, 'stix_id': technique['id']}
        })
    
    # Process Campaigns
    all_campaign_files = list((enterprise_dir / "campaign").glob("*.json"))
    filtered_campaign_files = filter_cti_files(all_campaign_files, "campaign")
    
    for file_path in filtered_campaign_files:
        data = load_json_file(str(file_path))
        campaign = data['objects'][0]
        
        mitre_id = next((ref['external_id'] for ref in campaign.get('external_references', [])
                        if ref.get('source_name') == 'mitre-attack'), None)
        
        properties = {
            'aliases': campaign.get('aliases', []),
            'created': to_neo4j_datetime(campaign.get('created')),
            'description': campaign.get('description', 'N/A'),
            'external_references': to_json_string(campaign.get('external_references', [])),
            'first_seen': to_neo4j_datetime(campaign.get('first_seen')),
            'id': campaign.get('id', 'N/A'),
            'last_seen': to_neo4j_datetime(campaign.get('last_seen')),
            'mitre_id': mitre_id or 'N/A',
            'modified': to_neo4j_datetime(campaign.get('modified')),
            'name': campaign.get('name', 'N/A'),
            'revoked': campaign.get('revoked', False),
            'type': campaign.get('type', 'N/A'),
            'x_mitre_deprecated': campaign.get('x_mitre_deprecated', False),
            'x_mitre_domains': campaign.get('x_mitre_domains', []),
            'x_mitre_version': campaign.get('x_mitre_version', 'N/A'),
            'x_mitre_attack_spec_version': campaign.get('x_mitre_attack_spec_version', 'N/A')
        }
        
        queries.append({
            'statement': create_node_query('campaign', campaign['id'], properties),
            'parameters': {**properties, 'stix_id': campaign['id']}
        })
    
    # Process Course of Action (Mitigation)
    all_coa_files = list((enterprise_dir / "course-of-action").glob("*.json"))
    filtered_coa_files = filter_cti_files(all_coa_files, "course-of-action")
    
    for file_path in filtered_coa_files:
        data = load_json_file(str(file_path))
        coa = data['objects'][0]
        
        mitre_id = next((ref['external_id'] for ref in coa.get('external_references', [])
                        if ref.get('source_name') == 'mitre-attack'), None)
        
        properties = {
            'created': to_neo4j_datetime(coa.get('created')),
            'description': coa.get('description', 'N/A'),
            'external_references': to_json_string(coa.get('external_references', [])),
            'id': coa.get('id', 'N/A'),
            'mitre_id': mitre_id or 'N/A',
            'modified': to_neo4j_datetime(coa.get('modified')),
            'name': coa.get('name', 'N/A'),
            'spec_version': coa.get('spec_version', 'N/A'),
            'type': coa.get('type', 'N/A'),
            'x_mitre_attack_spec_version': coa.get('x_mitre_attack_spec_version', 'N/A'),
            'x_mitre_deprecated': coa.get('x_mitre_deprecated', False),
            'x_mitre_domains': coa.get('x_mitre_domains', []),
            'x_mitre_modified_by_ref': coa.get('x_mitre_modified_by_ref', 'N/A'),
            'x_mitre_version': coa.get('x_mitre_version', 'N/A')
        }
        
        queries.append({
            'statement': create_node_query('course-of-action', coa['id'], properties),
            'parameters': {**properties, 'stix_id': coa['id']}
        })

    # Process Intrusion Sets
    all_intrusion_files = list((enterprise_dir / "intrusion-set").glob("*.json"))
    filtered_intrusion_files = filter_cti_files(all_intrusion_files, "intrusion-set")
    
    for file_path in filtered_intrusion_files:
        data = load_json_file(str(file_path))
        intrusion_set = data['objects'][0]
        
        mitre_id = next((ref['external_id'] for ref in intrusion_set.get('external_references', [])
                        if ref.get('source_name') == 'mitre-attack'), None)
        
        properties = {
            'aliases': intrusion_set.get('aliases', []),
            'created': to_neo4j_datetime(intrusion_set.get('created')),
            'description': intrusion_set.get('description', 'N/A'),
            'external_references': to_json_string(intrusion_set.get('external_references', [])),
            'id': intrusion_set.get('id', 'N/A'),
            'mitre_id': mitre_id or 'N/A',
            'modified': to_neo4j_datetime(intrusion_set.get('modified')),
            'name': intrusion_set.get('name', 'N/A'),
            'revoked': intrusion_set.get('revoked', False),
            'type': intrusion_set.get('type', 'N/A'),
            'x_mitre_deprecated': intrusion_set.get('x_mitre_deprecated', False),
            'x_mitre_domains': intrusion_set.get('x_mitre_domains', []),
            'x_mitre_version': intrusion_set.get('x_mitre_version', 'N/A'),
            'x_mitre_attack_spec_version': intrusion_set.get('x_mitre_attack_spec_version', 'N/A')
        }
        
        queries.append({
            'statement': create_node_query('intrusion-set', intrusion_set['id'], properties),
            'parameters': {**properties, 'stix_id': intrusion_set['id']}
        })
    
    # Process Malware
    all_malware_files = list((enterprise_dir / "malware").glob("*.json"))
    filtered_malware_files = filter_cti_files(all_malware_files, "malware")
    
    for file_path in filtered_malware_files:
        data = load_json_file(str(file_path))
        malware = data['objects'][0]
        
        mitre_id = next((ref['external_id'] for ref in malware.get('external_references', [])
                        if ref.get('source_name') == 'mitre-attack'), None)
        
        properties = {
            'created': to_neo4j_datetime(malware.get('created')),
            'description': malware.get('description', 'N/A'),
            'external_references': to_json_string(malware.get('external_references', [])),
            'id': malware.get('id', 'N/A'),
            'is_family': malware.get('is_family', False),
            'mitre_id': mitre_id or 'N/A',
            'modified': to_neo4j_datetime(malware.get('modified')),
            'name': malware.get('name', 'N/A'),
            'revoked': malware.get('revoked', False),
            'type': malware.get('type', 'N/A'),
            'x_mitre_aliases': malware.get('x_mitre_aliases', []),
            'x_mitre_deprecated': malware.get('x_mitre_deprecated', False),
            'x_mitre_domains': malware.get('x_mitre_domains', []),
            'x_mitre_platforms': malware.get('x_mitre_platforms', []),
            'x_mitre_version': malware.get('x_mitre_version', 'N/A'),
            'x_mitre_attack_spec_version': malware.get('x_mitre_attack_spec_version', 'N/A')
        }
        
        queries.append({
            'statement': create_node_query('malware', malware['id'], properties),
            'parameters': {**properties, 'stix_id': malware['id']}
        })
    
    # Process Tools
    all_tool_files = list((enterprise_dir / "tool").glob("*.json"))
    filtered_tool_files = filter_cti_files(all_tool_files, "tool")
    
    for file_path in filtered_tool_files:
        data = load_json_file(str(file_path))
        tool = data['objects'][0]
        
        mitre_id = next((ref['external_id'] for ref in tool.get('external_references', [])
                        if ref.get('source_name') == 'mitre-attack'), None)
        
        properties = {
            'created': to_neo4j_datetime(tool.get('created')),
            'created_by_ref': tool.get('created_by_ref', 'N/A'),
            'description': tool.get('description', 'N/A'),
            'external_references': to_json_string(tool.get('external_references', [])),
            'id': tool.get('id', 'N/A'),
            'mitre_id': mitre_id or 'N/A',
            'modified': to_neo4j_datetime(tool.get('modified')),
            'name': tool.get('name', 'N/A'),
            'revoked': tool.get('revoked', False),
            'type': tool.get('type', 'N/A'),
            'x_mitre_aliases': tool.get('x_mitre_aliases', []),
            'x_mitre_deprecated': tool.get('x_mitre_deprecated', False),
            'x_mitre_domains': tool.get('x_mitre_domains', []),
            'x_mitre_platforms': tool.get('x_mitre_platforms', []),
            'x_mitre_version': tool.get('x_mitre_version', 'N/A')
        }
        
        queries.append({
            'statement': create_node_query('tool', tool['id'], properties),
            'parameters': {**properties, 'stix_id': tool['id']}
        })

    # Process Data Components
    data_component_dir = enterprise_dir / "x-mitre-data-component"
    if data_component_dir.exists():
        all_dc_files = list(data_component_dir.glob("*.json"))
        filtered_dc_files = filter_cti_files(all_dc_files, "x-mitre-data-component")
        
        for file_path in filtered_dc_files:
            data = load_json_file(str(file_path))
            data_component = data['objects'][0]

            mitre_id = next((ref['external_id'] for ref in data_component.get('external_references', [])
                            if ref.get('source_name') == 'mitre-attack'), None)

            properties = {
                'created': to_neo4j_datetime(data_component.get('created')),
                'description': data_component.get('description', 'N/A'),
                'id': data_component.get('id', 'N/A'),
                'mitre_id': mitre_id or 'N/A',
                'modified': to_neo4j_datetime(data_component.get('modified')),
                'name': data_component.get('name', 'N/A'),
                'revoked': data_component.get('revoked', False),
                'type': data_component.get('type', 'N/A'),
                'x_mitre_deprecated': data_component.get('x_mitre_deprecated', False),
                'x_mitre_domains': data_component.get('x_mitre_domains', []),
                'x_mitre_version': data_component.get('x_mitre_version', 'N/A')
            }
            
            queries.append({
                'statement': create_node_query('x-mitre-data-component', data_component['id'], properties),
                'parameters': {**properties, 'stix_id': data_component['id']}
            })

    # Process Data Sources
    data_source_dir = enterprise_dir / "x-mitre-data-source"
    if data_source_dir.exists():
        all_ds_files = list(data_source_dir.glob("*.json"))
        filtered_ds_files = filter_cti_files(all_ds_files, "x-mitre-data-source")

        for file_path in filtered_ds_files:
            data = load_json_file(str(file_path))
            data_source = data['objects'][0]

            mitre_id = next((ref['external_id'] for ref in data_source.get('external_references', [])
                            if ref.get('source_name') == 'mitre-attack'), None)

            properties = {
                'created': to_neo4j_datetime(data_source.get('created')),
                'description': data_source.get('description', 'N/A'),
                'id': data_source.get('id', 'N/A'),
                'mitre_id': mitre_id or 'N/A',
                'modified': to_neo4j_datetime(data_source.get('modified')),
                'name': data_source.get('name', 'N/A'),
                'revoked': data_source.get('revoked', False),
                'type': data_source.get('type', 'N/A'),
                'x_mitre_deprecated': data_source.get('x_mitre_deprecated', False),
                'x_mitre_domains': data_source.get('x_mitre_domains', []),
                'x_mitre_version': data_source.get('x_mitre_version', 'N/A'),
                'x_mitre_attack_spec_version': data_source.get('x_mitre_attack_spec_version', 'N/A')
            }

            queries.append({
                'statement': create_node_query('x-mitre-data-source', data_source['id'], properties),
                'parameters': {**properties, 'stix_id': data_source['id']}
            })
    
    # Process Tactics
    all_tactic_files = list((enterprise_dir / "x-mitre-tactic").glob("*.json"))
    filtered_tactic_files = filter_cti_files(all_tactic_files, "x-mitre-tactic")
    
    for file_path in filtered_tactic_files:
        data = load_json_file(str(file_path))
        tactic = data['objects'][0]
        
        mitre_id = next((ref['external_id'] for ref in tactic.get('external_references', [])
                        if ref.get('source_name') == 'mitre-attack'), None)
        
        properties = {
            'created': to_neo4j_datetime(tactic.get('created')),
            'description': tactic.get('description', 'N/A'),
            'external_references': to_json_string(tactic.get('external_references', [])),
            'id': tactic.get('id', 'N/A'),
            'mitre_id': mitre_id or 'N/A',
            'modified': to_neo4j_datetime(tactic.get('modified')),
            'name': tactic.get('name', 'N/A'),
            'type': tactic.get('type', 'N/A'),
            'x_mitre_deprecated': tactic.get('x_mitre_deprecated', False),
            'x_mitre_domains': tactic.get('x_mitre_domains', []),
            'x_mitre_shortname': tactic.get('x_mitre_shortname', 'N/A'),
            'x_mitre_version': tactic.get('x_mitre_version', 'N/A'),
            'x_mitre_attack_spec_version': tactic.get('x_mitre_attack_spec_version', 'N/A')
        }
        
        queries.append({
            'statement': create_node_query('x-mitre-tactic', tactic['id'], properties),
            'parameters': {**properties, 'stix_id': tactic['id']}
        })

    # Process Analytics
    analytic_dir = enterprise_dir / "x-mitre-analytic"
    if analytic_dir.exists():
        all_analytic_files = list(analytic_dir.glob("*.json"))
        filtered_analytic_files = filter_cti_files(all_analytic_files, "x-mitre-analytic")

        for file_path in filtered_analytic_files:
            data = load_json_file(str(file_path))
            analytic = data['objects'][0]

            mitre_id = next((ref['external_id'] for ref in analytic.get('external_references', [])
                            if ref.get('source_name') == 'mitre-attack'), None)

            properties = {
                'created': to_neo4j_datetime(analytic.get('created')),
                'created_by_ref': analytic.get('created_by_ref', 'N/A'),
                'description': analytic.get('description', 'N/A'),
                'external_references': to_json_string(analytic.get('external_references', [])),
                'id': analytic.get('id', 'N/A'),
                'mitre_id': mitre_id or 'N/A',
                'modified': to_neo4j_datetime(analytic.get('modified')),
                'name': analytic.get('name', 'N/A'),
                'type': analytic.get('type', 'N/A'),
                'x_mitre_deprecated': analytic.get('x_mitre_deprecated', False),
                'x_mitre_domains': analytic.get('x_mitre_domains', []),
                'x_mitre_platforms': analytic.get('x_mitre_platforms', []),
                'x_mitre_version': analytic.get('x_mitre_version', 'N/A'),
                'x_mitre_attack_spec_version': analytic.get('x_mitre_attack_spec_version', 'N/A'),
                'x_mitre_log_source_references': to_json_string(analytic.get('x_mitre_log_source_references', [])),
                'x_mitre_mutable_elements': to_json_string(analytic.get('x_mitre_mutable_elements', []))
            }

            queries.append({
                'statement': create_node_query('x-mitre-analytic', analytic['id'], properties),
                'parameters': {**properties, 'stix_id': analytic['id']}
            })

    # Process Detection Strategies
    detection_strategy_dir = enterprise_dir / "x-mitre-detection-strategy"
    if detection_strategy_dir.exists():
        all_ds_strat_files = list(detection_strategy_dir.glob("*.json"))
        filtered_ds_strat_files = filter_cti_files(all_ds_strat_files, "x-mitre-detection-strategy")

        for file_path in filtered_ds_strat_files:
            data = load_json_file(str(file_path))
            detection_strategy = data['objects'][0]

            mitre_id = next((ref['external_id'] for ref in detection_strategy.get('external_references', [])
                            if ref.get('source_name') == 'mitre-attack'), None)

            properties = {
                'created': to_neo4j_datetime(detection_strategy.get('created')),
                'created_by_ref': detection_strategy.get('created_by_ref', 'N/A'),
                'external_references': to_json_string(detection_strategy.get('external_references', [])),
                'id': detection_strategy.get('id', 'N/A'),
                'mitre_id': mitre_id or 'N/A',
                'modified': to_neo4j_datetime(detection_strategy.get('modified')),
                'name': detection_strategy.get('name', 'N/A'),
                'type': detection_strategy.get('type', 'N/A'),
                'x_mitre_deprecated': detection_strategy.get('x_mitre_deprecated', False),
                'x_mitre_domains': detection_strategy.get('x_mitre_domains', []),
                'x_mitre_analytic_refs': detection_strategy.get('x_mitre_analytic_refs', []),
                'x_mitre_version': detection_strategy.get('x_mitre_version', 'N/A'),
                'x_mitre_attack_spec_version': detection_strategy.get('x_mitre_attack_spec_version', 'N/A')
            }

            queries.append({
                'statement': create_node_query('x-mitre-detection-strategy', detection_strategy['id'], properties),
                'parameters': {**properties, 'stix_id': detection_strategy['id']}
            })
    
    return queries

def process_capec_data(xml_file: Path) -> List[Dict]:
    """Process CAPEC data from XML file."""
    queries = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define namespace
    ns = {'capec': 'http://capec.mitre.org/capec-3'}

    def get_text(elem, path):
        """Return normalized text of the first matching subelement, or ''."""
        found = elem.find(path, ns)
        if found is not None:
            return ' '.join(''.join(found.itertext()).split())
        return ''

    def elem_to_dict(elem):
        """Recursively convert an XML element to a plain dict (strips namespace URIs)."""
        children = list(elem)
        if not children:
            return (elem.text or '').strip()
        d = {}
        if elem.attrib:
            d['@attributes'] = dict(elem.attrib)
        for child in children:
            child_tag = child.tag.split('}')[-1]
            child_val = elem_to_dict(child)
            if child_tag in d:
                if not isinstance(d[child_tag], list):
                    d[child_tag] = [d[child_tag]]
                d[child_tag].append(child_val)
            else:
                d[child_tag] = child_val
        return d

    # Process attack patterns
    for pattern in root.findall('.//capec:Attack_Pattern', ns):
        capec_id = pattern.get('ID')
        if not capec_id:
            continue

        # Generate STIX ID
        stix_id = generate_stix_id('capec', f"CAPEC-{capec_id}")

        # --- XML attributes ---
        abstraction = pattern.get('Abstraction', '')
        name = pattern.get('Name', '')
        status = pattern.get('Status', '')

        # --- description ---
        description = get_text(pattern, 'capec:Description')

        # --- likelihood_of_attack ---
        likelihood_of_attack = get_text(pattern, 'capec:Likelihood_Of_Attack')

        # --- severity ---
        severity = get_text(pattern, 'capec:Typical_Severity')

        # --- consequences: "Scope: S1, S2; Impact: I" ---
        consequences = []
        consequences_elem = pattern.find('capec:Consequences', ns)
        if consequences_elem is not None:
            for consequence in consequences_elem.findall('capec:Consequence', ns):
                scopes = [s.text.strip() for s in consequence.findall('capec:Scope', ns) if s.text]
                impact_elem = consequence.find('capec:Impact', ns)
                impact = impact_elem.text.strip() if impact_elem is not None and impact_elem.text else ''
                entry = f"Scope: {', '.join(scopes)}; Impact: {impact}"
                consequences.append(entry)

        # --- indicators ---
        indicators = []
        indicators_elem = pattern.find('capec:Indicators', ns)
        if indicators_elem is not None:
            indicators = [' '.join(''.join(ind.itertext()).split())
                          for ind in indicators_elem.findall('capec:Indicator', ns)
                          if ''.join(ind.itertext()).strip()]

        # --- example_instances ---
        example_instances = []
        examples_elem = pattern.find('capec:Example_Instances', ns)
        if examples_elem is not None:
            example_instances = [' '.join(''.join(ex.itertext()).split())
                                  for ex in examples_elem.findall('capec:Example', ns)
                                  if ''.join(ex.itertext()).strip()]

        # --- mitigations ---
        mitigations = []
        mitigations_elem = pattern.find('capec:Mitigations', ns)
        if mitigations_elem is not None:
            mitigations = [' '.join(''.join(m.itertext()).split())
                           for m in mitigations_elem.findall('capec:Mitigation', ns)
                           if ''.join(m.itertext()).strip()]

        # --- prerequisites ---
        prerequisites = []
        prereq_elem = pattern.find('capec:Prerequisites', ns)
        if prereq_elem is not None:
            prerequisites = [' '.join(''.join(p.itertext()).split())
                             for p in prereq_elem.findall('capec:Prerequisite', ns)
                             if ''.join(p.itertext()).strip()]

        # --- skills_required: "Level: {Level}. {text}" ---
        skills_required = []
        skills_elem = pattern.find('capec:Skills_Required', ns)
        if skills_elem is not None:
            for skill in skills_elem.findall('capec:Skill', ns):
                level = skill.get('Level', '')
                text = ' '.join(''.join(skill.itertext()).split())
                skills_required.append(f"Level: {level}. {text}")

        # --- created (Submission_Date) ---
        created_str = get_text(pattern, './/capec:Content_History/capec:Submission/capec:Submission_Date')
        created = to_neo4j_datetime(created_str) if created_str else 'N/A'

        # --- modified (last Modification_Date) ---
        modifications = pattern.findall('.//capec:Content_History/capec:Modification', ns)
        modified = 'N/A'
        if modifications:
            mod_date_elem = modifications[-1].find('capec:Modification_Date', ns)
            if mod_date_elem is not None and mod_date_elem.text:
                modified = to_neo4j_datetime(mod_date_elem.text.strip())

        # --- execution_flow (JSON string) ---
        execution_flow = 'N/A'
        ef_elem = pattern.find('capec:Execution_Flow', ns)
        if ef_elem is not None:
            execution_flow = json.dumps(elem_to_dict(ef_elem))

        properties = {
            'abstraction': abstraction,
            'id': int(capec_id),
            'capec_id': f"CAPEC-{capec_id}",
            'name': name,
            'status': status,
            'consequences': consequences,
            'created': created,
            'indicators': indicators,
            'modified': modified,
            'description': description,
            'example_instances': example_instances,
            'likelihood_of_attack': likelihood_of_attack,
            'mitigations': mitigations,
            'prerequisites': prerequisites,
            'skills_required': skills_required,
            'severity': severity,
            'execution_flow': execution_flow,
        }
        queries.append({
            'statement': create_node_query('CAPEC', stix_id, properties),
            'parameters': {**properties, 'stix_id': stix_id}
        })
    return queries

def create_capec_internal_relationships(xml_file: Path) -> List[Dict]:
    """Create intra-CAPEC relationships from CAPEC XML data."""
    queries = []
    
    logger.info("Creating intra-CAPEC relationships...")
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'capec': 'http://capec.mitre.org/capec-3'}
    
    relationship_counts = {}

    for pattern in root.findall('.//capec:Attack_Pattern', ns):
        source_capec_id = pattern.get('ID')
        if not source_capec_id:
            continue

        source_stix_id = generate_stix_id('capec', f"CAPEC-{source_capec_id}")

        # Process Related_Attack_Patterns
        related_patterns = pattern.find('.//capec:Related_Attack_Patterns', ns)
        if related_patterns is not None:
            for related_pattern in related_patterns.findall('.//capec:Related_Attack_Pattern', ns):
                # Use the raw @Nature value as the relationship type (e.g. ChildOf, CanPrecede, PeerOf)
                nature = related_pattern.get('Nature', '').strip()
                target_capec_id = related_pattern.get('CAPEC_ID')

                if not nature or not target_capec_id:
                    continue

                target_stix_id = generate_stix_id('capec', f"CAPEC-{target_capec_id}")

                query = f"""
                MATCH (c1:CAPEC {{stix_id: '{source_stix_id}'}})
                MATCH (c2:CAPEC {{stix_id: '{target_stix_id}'}})
                MERGE (c1)-[r:{nature}]->(c2)
                """
                queries.append({'statement': query})
                relationship_counts[nature] = relationship_counts.get(nature, 0) + 1

    # Log relationship counts
    for rel_type, count in relationship_counts.items():
        logger.info(f"Created {count} {rel_type} relationships")

    total_relationships = sum(relationship_counts.values())
    logger.info(f"Total intra-CAPEC relationships created: {total_relationships}")
    
    return queries

def process_cwe_data(xml_file: Path) -> List[Dict]:
    """Process CWE data from XML file - creates Weakness, Detection_Method, Mitigation nodes
    and their relationships. Structured in passes: nodes first, then relationships."""
    queries = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}

    def clean_text(elem):
        """Return normalized whitespace text from element via itertext."""
        if elem is None:
            return 'N/A'
        return ' '.join(''.join(elem.itertext()).split()) or 'N/A'

    weaknesses = root.findall('.//cwe:Weakness', ns)

    # ================================================================
    # PASS 1: Weakness nodes
    # ================================================================
    for weakness in weaknesses:
        cwe_id = weakness.get('ID')
        if not cwe_id:
            continue
        weakness_id = f"CWE-{cwe_id}"

        name         = weakness.get('Name', 'N/A')
        abstraction  = weakness.get('Abstraction', 'N/A')
        diagram      = weakness.get('Diagram', 'N/A')
        description  = clean_text(weakness.find('cwe:Description', ns))
        extended_description  = clean_text(weakness.find('cwe:Extended_Description', ns))
        likelihood_of_exploit = clean_text(weakness.find('cwe:Likelihood_Of_Exploit', ns))

        modifications = weakness.findall('.//cwe:Content_History/cwe:Modification', ns)
        modified = 'N/A'
        if modifications:
            mod_date_elem = modifications[-1].find('cwe:Modification_Date', ns)
            if mod_date_elem is not None and mod_date_elem.text:
                modified = to_neo4j_datetime(mod_date_elem.text.strip())

        functional_areas = [fa.text.strip()
                            for fa in weakness.findall('.//cwe:Functional_Areas/cwe:Functional_Area', ns)
                            if fa.text]

        intro_list = []
        for intro in weakness.findall('.//cwe:Modes_Of_Introduction/cwe:Introduction', ns):
            phase_elem = intro.find('cwe:Phase', ns)
            note_elem  = intro.find('cwe:Note', ns)
            intro_list.append({
                'phase': phase_elem.text.strip() if phase_elem is not None and phase_elem.text else '',
                'note':  clean_text(note_elem) if note_elem is not None else ''
            })
        modes_of_introduction = to_json_string(intro_list)
        phases = [entry['phase'] for entry in intro_list if entry['phase']]

        stix_id = generate_stix_id('cwe', weakness_id)
        properties = {
            'id': weakness_id,
            'name': name,
            'abstraction': abstraction,
            'diagram': diagram,
            'description': description,
            'extended_description': extended_description,
            'likelihood_of_exploit': likelihood_of_exploit,
            'modified': modified,
            'functional_areas': functional_areas,
            'modes_of_introduction': modes_of_introduction,
            'phases': phases,
        }
        queries.append({
            'statement': create_node_query('Weakness', stix_id, properties),
            'parameters': {**properties, 'stix_id': stix_id}
        })

    # ================================================================
    # PASS 2: Detection_Method nodes (deduplicated across all Weaknesses)
    # ================================================================
    detection_methods_seen = set()

    for weakness in weaknesses:
        cwe_id = weakness.get('ID')
        if not cwe_id:
            continue

        for det_method in weakness.findall('.//cwe:Detection_Methods/cwe:Detection_Method', ns):
            det_id = det_method.get('Detection_Method_ID', '').strip()
            method = clean_text(det_method.find('cwe:Method', ns))
            det_description = clean_text(det_method.find('cwe:Description', ns))
            effectiveness   = clean_text(det_method.find('cwe:Effectiveness', ns))
            if not det_id:
                det_id = f"DM_{method}"
            if det_id not in detection_methods_seen:
                detection_methods_seen.add(det_id)
                det_stix_id = generate_stix_id('cwe', det_id)
                det_props = {
                    'id': det_id,
                    'method': method,
                    'description': det_description,
                    'effectiveness': effectiveness,
                }
                queries.append({
                    'statement': create_node_query('Detection_Method', det_stix_id, det_props),
                    'parameters': {**det_props, 'stix_id': det_stix_id}
                })

    # ================================================================
    # PASS 3: Observed_Example nodes (deduplicated by reference/CVE ID)
    # ================================================================
    observed_examples_seen = set()

    for weakness in weaknesses:
        if not weakness.get('ID'):
            continue
        for obs in weakness.findall('.//cwe:Observed_Examples/cwe:Observed_Example', ns):
            reference = clean_text(obs.find('cwe:Reference', ns))
            if reference == 'N/A' or reference in observed_examples_seen:
                continue
            observed_examples_seen.add(reference)
            link            = clean_text(obs.find('cwe:Link', ns))
            obs_description = clean_text(obs.find('cwe:Description', ns))
            oe_stix_id = generate_stix_id('cwe', reference)
            oe_props = {
                'id': reference,
                'reference': reference,
                'link': link,
                'description': obs_description,
            }
            queries.append({
                'statement': create_node_query('Observed_Example', oe_stix_id, oe_props),
                'parameters': {**oe_props, 'stix_id': oe_stix_id}
            })

    # ================================================================
    # PASS 4: Mitigation nodes
    # ================================================================
    for weakness in weaknesses:
        cwe_id = weakness.get('ID')
        if not cwe_id:
            continue
        weakness_id = f"CWE-{cwe_id}"

        for mitigation in weakness.findall('.//cwe:Potential_Mitigations/cwe:Mitigation', ns):
            mit_id = mitigation.get('Mitigation_ID', '')
            mit_phases = [p.text.strip() for p in mitigation.findall('cwe:Phase', ns) if p.text]
            mit_description = clean_text(mitigation.find('cwe:Description', ns))
            effectiveness   = clean_text(mitigation.find('cwe:Effectiveness', ns))
            strategy        = clean_text(mitigation.find('cwe:Strategy', ns))
            # Always scope by weakness_id — MIT-1 repeats across weaknesses
            raw_mit_id    = mit_id if mit_id else f"MIT_{'_'.join(mit_phases)}"
            mitigation_id = f"{weakness_id}_{raw_mit_id}"
            mit_stix_id   = generate_stix_id('cwe', mitigation_id)
            mit_props = {
                'id': mitigation_id,
                'phase': mit_phases,
                'description': mit_description,
                'effectiveness': effectiveness,
                'strategy': strategy,
            }
            queries.append({
                'statement': create_node_query('Mitigation', mit_stix_id, mit_props),
                'parameters': {**mit_props, 'stix_id': mit_stix_id}
            })

    # ================================================================
    # PASS 5: Relationships
    # ================================================================
    for weakness in weaknesses:
        cwe_id = weakness.get('ID')
        if not cwe_id:
            continue
        weakness_id = f"CWE-{cwe_id}"

        # Weakness -[:observed_as {description, link}]-> CVE
        for obs in weakness.findall('.//cwe:Observed_Examples/cwe:Observed_Example', ns):
            reference    = clean_text(obs.find('cwe:Reference', ns))
            link         = clean_text(obs.find('cwe:Link', ns))
            obs_desc     = clean_text(obs.find('cwe:Description', ns))
            if reference == 'N/A':
                continue
            queries.append({
                'statement': (
                    "MATCH (w:Weakness {id: $weakness_id}) "
                    "MATCH (c:CVE {id: $reference}) "
                    "MERGE (w)-[r:observed_as]->(c) "
                    "SET r.description = $description, r.link = $link"
                ),
                'parameters': {
                    'weakness_id': weakness_id,
                    'reference': reference,
                    'description': obs_desc,
                    'link': link,
                }
            })

        # Weakness -[:related_attack_pattern]-> CAPEC
        for rap in weakness.findall('.//cwe:Related_Attack_Patterns/cwe:Related_Attack_Pattern', ns):
            capec_id = rap.get('CAPEC_ID', '')
            if capec_id:
                capec_stix_id = generate_stix_id('capec', f"CAPEC-{capec_id}")
                queries.append({
                    'statement': (
                        "MATCH (w:Weakness {id: $weakness_id}) "
                        "MATCH (c:CAPEC {stix_id: $capec_stix_id}) "
                        "MERGE (w)-[:related_attack_pattern]->(c)"
                    ),
                    'parameters': {'weakness_id': weakness_id, 'capec_stix_id': capec_stix_id}
                })

        # Weakness -[:mitigated_by]-> Mitigation
        for mitigation in weakness.findall('.//cwe:Potential_Mitigations/cwe:Mitigation', ns):
            mit_id = mitigation.get('Mitigation_ID', '')
            mit_phases = [p.text.strip() for p in mitigation.findall('cwe:Phase', ns) if p.text]
            raw_mit_id    = mit_id if mit_id else f"MIT_{'_'.join(mit_phases)}"
            mitigation_id = f"{weakness_id}_{raw_mit_id}"
            queries.append({
                'statement': (
                    "MATCH (w:Weakness {id: $weakness_id}) "
                    "MATCH (m:Mitigation {id: $mitigation_id}) "
                    "MERGE (w)-[:mitigated_by]->(m)"
                ),
                'parameters': {'weakness_id': weakness_id, 'mitigation_id': mitigation_id}
            })

        # Weakness -[:detected_by]-> Detection_Method
        for det_method in weakness.findall('.//cwe:Detection_Methods/cwe:Detection_Method', ns):
            det_id = det_method.get('Detection_Method_ID', '').strip()
            if not det_id:
                method = clean_text(det_method.find('cwe:Method', ns))
                det_id = f"DM_{method}"
            queries.append({
                'statement': (
                    "MATCH (w:Weakness {id: $weakness_id}) "
                    "MATCH (d:Detection_Method {id: $det_id}) "
                    "MERGE (w)-[:detected_by]->(d)"
                ),
                'parameters': {'weakness_id': weakness_id, 'det_id': det_id}
            })

        # CWE internal relationships: ChildOf, CanPrecede, PeerOf
        for rel in weakness.findall('.//cwe:Related_Weaknesses/cwe:Related_Weakness', ns):
            rel_nature = rel.get('Nature', '')
            rel_cwe_id = rel.get('CWE_ID', '')
            if rel_cwe_id and rel_nature in ['ChildOf', 'CanPrecede', 'PeerOf']:
                target_id = f"CWE-{rel_cwe_id}"
                queries.append({
                    'statement': (
                        f"MATCH (w1:Weakness {{id: $w1_id}}) "
                        f"MATCH (w2:Weakness {{id: $w2_id}}) "
                        f"MERGE (w1)-[:{rel_nature}]->(w2)"
                    ),
                    'parameters': {'w1_id': weakness_id, 'w2_id': target_id}
                })

    logger.info(f"Processed CWE data: {len(detection_methods_seen)} unique Detection_Methods, "
                f"{len(observed_examples_seen)} unique Observed_Examples")
    return queries

def _build_nvd_cpe_index(nvd_dir: Path) -> Dict[str, List]:
    """Load CVE-ID → CPE match list from NVD NDJSON batch files."""
    nvd_index: Dict[str, List] = {}
    if not nvd_dir.exists():
        logger.warning(f"NVD directory not found at {nvd_dir}, CVE nodes will have no cpe_matches data")
        return nvd_index
    for year_dir in sorted(nvd_dir.iterdir()):
        if not year_dir.is_dir():
            continue
        for batch_file in sorted(year_dir.glob("batch_*.ndjson")):
            try:
                with batch_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        cve_obj = entry.get("cve", {})
                        cve_id = cve_obj.get("id")
                        if not cve_id:
                            continue

                        matches = [
                            {k: v for k, v in m.items() if k in (
                                "criteria", "vulnerable",
                                "versionStartIncluding", "versionStartExcluding",
                                "versionEndIncluding", "versionEndExcluding",
                            )}
                            for ng in cve_obj.get("configurations", [])
                            for node in ng.get("nodes", [])
                            for m in node.get("cpeMatch", [])
                            if m.get("criteria")
                        ]

                        if matches:
                            nvd_index[cve_id] = matches
            except Exception as e:
                logger.error(f"Failed to load NVD batch {batch_file}: {e}")
    logger.info(f"NVD index built: {len(nvd_index)} CVEs with CPE/CVSS data")
    return nvd_index


def process_cve_data(cve_dir: Path) -> List[Dict]:
    """Process CVE data from JSON files - using id (cveId) as key."""
    queries = []

    # Load NVD CPE index from sibling nvd/ directory
    nvd_dir = cve_dir.parent / "nvd"
    nvd_index = _build_nvd_cpe_index(nvd_dir)

    # Get all JSON files and filter out ignored ones
    all_json_files = list(cve_dir.rglob("*.json"))
    filtered_json_files = filter_cve_files(all_json_files)
    
    logger.info(f"Processing CVE data: {len(all_json_files)} total files, {len(filtered_json_files)} after filtering")
    
    # Process filtered JSON files in the CVE directory
    processed_count = 0
    for idx, json_file in enumerate(filtered_json_files, 1):
        try:
            data = load_json_file(str(json_file))
             
            if data.get('dataType') == 'CVE_RECORD':
                cve_metadata = data.get('cveMetadata', {})
                cve_id = cve_metadata.get('cveId')
                
                if cve_id:
                    containers = data.get('containers', {})
                    cna = containers.get('cna', {})

                    # title
                    title = cna.get('title', 'N/A')

                    # descriptions → list[str] of all values
                    descriptions = [
                        d.get('value', '')
                        for d in cna.get('descriptions', [])
                        if d.get('value')
                    ]

                    # affected → JSON string + affected_products list[str]
                    affected_raw = cna.get('affected', [])
                    affected = to_json_string(affected_raw)
                    affected_products = [
                        a.get('product', '')
                        for a in affected_raw
                        if a.get('product')
                    ]

                    # metrics → JSON string + top-level cvssV3_1 scalars
                    # Fall back to NVD cvssMetricV31 data if fields are missing from CVE 5.0 record
                    metrics_raw = cna.get('metrics', [])
                    metrics = to_json_string(metrics_raw)
                    cvss31 = metrics_raw[0].get('cvssV3_1', {}) if metrics_raw else {}
                    cvss3_1_attackVector = cvss31.get('attackVector', 'N/A')
                    cvss3_1_baseScore    = cvss31.get('baseScore', None)
                    cvss3_1_vectorString = cvss31.get('vectorString', None)

                    # workarounds / exploits → list[str] from .description field
                    workarounds = [
                        w.get('description', '')
                        for w in cna.get('workarounds', [])
                        if w.get('description')
                    ]
                    exploits = [
                        e.get('description', '')
                        for e in cna.get('exploits', [])
                        if e.get('description')
                    ]

                    # problemTypes kept internally for relationship creation only
                    problem_types = cna.get('problemTypes', [])

                    stix_id = generate_stix_id('cve', cve_id)
                    properties = {
                        'id': cve_id,
                        'created': to_neo4j_datetime(cve_metadata.get('datePublished')),
                        'modified': to_neo4j_datetime(cve_metadata.get('dateUpdated')),
                        'title': title,
                        'descriptions': descriptions,
                        'affected': affected,
                        'affected_products': affected_products,
                        'metrics': metrics,
                        'cvss3_1_attackVector': cvss3_1_attackVector,
                        'cvss3_1_baseScore': cvss3_1_baseScore,
                        'cvss3_1_vectorString': cvss3_1_vectorString,
                        'workarounds': workarounds,
                        'exploits': exploits,
                        'cpe_matches': to_json_string(nvd_index.get(cve_id, [])),
                    }
                    queries.append({
                        'statement': create_node_query('CVE', stix_id, properties),
                        'parameters': {**properties, 'stix_id': stix_id}
                    })
                    
                    # CVE -[:impacts]-> CAPEC (direct from containers/cna/impacts/[]/capecId)
                    for impact in cna.get('impacts', []):
                        capec_id = impact.get('capecId', '').strip()
                        if capec_id:
                            # capecId may come as "CAPEC-79" or just "79"
                            if not capec_id.startswith('CAPEC-'):
                                capec_id = f"CAPEC-{capec_id}"
                            capec_stix_id = generate_stix_id('capec', capec_id)
                            queries.append({
                                'statement': (
                                    "MATCH (c:CVE {id: $cve_id}) "
                                    "MATCH (cap:CAPEC {stix_id: $capec_stix_id}) "
                                    "MERGE (c)-[:impacts]->(cap)"
                                ),
                                'parameters': {'cve_id': cve_id, 'capec_stix_id': capec_stix_id}
                            })

                    # Create CVE-[:problemType]->Weakness and Weakness-[:observed_as]->CVE relationships
                    for problem in problem_types:
                        for desc in problem.get('descriptions', []):
                            cwe_id = desc.get('cweId', '')
                            desc_text = desc.get('description', 'N/A')
                            if cwe_id:
                                # CVE -> Weakness (problemType)
                                queries.append({
                                    'statement': """
                                    MATCH (c:CVE {id: $cve_id})
                                    MATCH (w:Weakness {id: $cwe_id})
                                    MERGE (c)-[:problemType]->(w)
                                    """,
                                    'parameters': {
                                        'cve_id': cve_id,
                                        'cwe_id': cwe_id
                                    }
                                })
                                # Weakness -> CVE (observed_as with Description)
                                queries.append({
                                    'statement': """
                                    MATCH (w:Weakness {id: $cwe_id})
                                    MATCH (c:CVE {id: $cve_id})
                                    MERGE (w)-[r:observed_as]->(c)
                                    SET r.description = $description
                                    """,
                                    'parameters': {
                                        'cwe_id': cwe_id,
                                        'cve_id': cve_id,
                                        'description': desc_text[:200] if desc_text else 'N/A'
                                    }
                                })
                    
                    processed_count += 1

        except Exception as e:
            logger.warning(f"Error processing CVE file {json_file}: {e}")
            continue
    
        # Log progress every 10,000 files
        if idx % 10000 == 0:
            logger.info(f"Progress: Processed {idx:,} / {len(filtered_json_files):,} CVE files ({processed_count:,} valid CVEs)")
    
    logger.info(f"Processed {processed_count:,} CVE entries from {len(filtered_json_files):,} files")
    return queries

def process_kev_data(kev_file: Path) -> List[Dict]:
    """Process KEV data from JSON file."""
    queries = []
    logger.info("Processing KEV data...")
    try:
        data = load_json_file(str(kev_file))
        if "vulnerabilities" not in data:
            logger.error("KEV data missing 'vulnerabilities' field")
            return queries
        vulnerabilities = data["vulnerabilities"]
        for vuln in vulnerabilities:
            try:
                cve_id = vuln.get("cveID", "")
                if not cve_id:
                    continue
                stix_id = generate_stix_id('kev', cve_id)
                properties = {
                    'cveID': cve_id,
                    'vendorProject': vuln.get('vendorProject', 'N/A'),
                    'product': vuln.get('product', 'N/A'),
                    'vulnerabilityName': vuln.get('vulnerabilityName', 'N/A'),
                    'dateAdded': to_neo4j_datetime(vuln.get('dateAdded', 'N/A')),
                    'shortDescription': vuln.get('shortDescription', 'N/A'),
                    'requiredAction': vuln.get('requiredAction', 'N/A'),
                    'dueDate': to_neo4j_datetime(vuln.get('dueDate', 'N/A')),
                    'knownRansomwareCampaignUse': vuln.get('knownRansomwareCampaignUse', 'N/A'),
                    'notes': vuln.get('notes', 'N/A'),
                    'cwes': [
                        e.get('cweId', '').strip() if isinstance(e, dict) else str(e).strip()
                        for e in vuln.get('cwes', [])
                        if (e.get('cweId') if isinstance(e, dict) else e)
                    ],
                }

                queries.append({
                    'statement': create_node_query('KEV', stix_id, properties),
                    'parameters': {**properties, 'stix_id': stix_id}
                })

                # KEV -[:for_CVE]-> CVE (multiplicity 1, defined by KEV.cveId)
                queries.append({
                    'statement': (
                        "MATCH (k:KEV {stix_id: $stix_id}) "
                        "MATCH (c:CVE {id: $cve_id}) "
                        "MERGE (k)-[:for_CVE]->(c)"
                    ),
                    'parameters': {'stix_id': stix_id, 'cve_id': cve_id}
                })

                # CVE -[:known_exploit]-> KEV (multiplicity 1, defined by KEV.cveId)
                queries.append({
                    'statement': (
                        "MATCH (c:CVE {id: $cve_id}) "
                        "MATCH (k:KEV {stix_id: $stix_id}) "
                        "MERGE (c)-[:known_exploit]->(k)"
                    ),
                    'parameters': {'cve_id': cve_id, 'stix_id': stix_id}
                })

                # Weakness -[:known_exploit]-> KEV (driven by KEV.cwes list)
                for cwe_entry in vuln.get('cwes', []):
                    if isinstance(cwe_entry, dict):
                        cwe_id_str = cwe_entry.get('cweId', '').strip()
                    else:
                        cwe_id_str = cwe_entry.strip() if isinstance(cwe_entry, str) else ''
                    if cwe_id_str:
                        queries.append({
                            'statement': (
                                "MATCH (w:Weakness {id: $cwe_id}) "
                                "MATCH (k:KEV {stix_id: $stix_id}) "
                                "MERGE (w)-[:known_exploit]->(k)"
                            ),
                            'parameters': {'cwe_id': cwe_id_str, 'stix_id': stix_id}
                        })

            except Exception as e:
                import traceback
                logger.error(f"Error processing KEV entry {vuln.get('cveID', 'unknown')}: {e}")
                traceback.print_exc()
                continue

        logger.info(f"Processed {len(queries)} KEV entries")

    except Exception as e:
        import traceback
        logger.error(f"Error processing KEV file {kev_file}: {e}")
        traceback.print_exc()
        
    return queries

def download_epss_data(data_dir: Path) -> Path:
    """Download yesterday's EPSS scores and overwrite epss_scores_data.csv."""
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    url = DATA_SOURCES["epss"].format(date=yesterday)
    epss_file = data_dir / "epss_scores_data.csv"
    logger.info(f"Downloading EPSS data for {yesterday}...")
    response = requests.get(url)
    response.raise_for_status()
    with gzip.open(io.BytesIO(response.content), "rt") as f_in:
        lines = f_in.readlines()
    header = next(l for l in lines if not l.startswith("#"))
    data_lines = [
        l for l in lines
        if not l.startswith("#") and not l.startswith("cve,")
        and any(l.startswith(f"CVE-{year}") for year in range(2024, datetime.today().year + 1))
    ]
    with open(epss_file, "w") as f_out:
        f_out.write(header)
        f_out.writelines(data_lines)
    logger.info(f"EPSS data saved to: {epss_file}")
    return epss_file


def process_epss_data(csv_file: Path) -> List[Dict]:
    """Process EPSS scores CSV and create EPSS nodes linked to CVE nodes."""
    queries = []

    if not csv_file.exists():
        logger.warning(f"EPSS CSV not found: {csv_file}")
        return queries

    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cve_id = row.get("cve", "").strip()
                if not cve_id:
                    continue
                # Only process CVE-2024 and newer
                try:
                    cve_year = int(cve_id.split("-")[1])
                    if cve_year < 2024:
                        continue
                except (IndexError, ValueError):
                    continue
                epss_score = row.get("epss", "N/A").strip()
                percentile = row.get("percentile", "N/A").strip()
                stix_id = generate_stix_id("epss", cve_id)
                properties = {
                    "cve": cve_id,
                    "epss": float(epss_score) if epss_score != "N/A" else None,
                    "percentile": float(percentile) if percentile != "N/A" else None,
                }
                queries.append({
                    "statement": create_node_query("EPSS", stix_id, properties),
                    "parameters": {**properties, "stix_id": stix_id}
                })
                # EPSS -[:scores]-> CVE
                queries.append({
                    "statement": (
                        "MATCH (e:EPSS {stix_id: $stix_id}) "
                        "MATCH (c:CVE {id: $cve_id}) "
                        "MERGE (e)-[:scores]->(c)"
                    ),
                    "parameters": {"stix_id": stix_id, "cve_id": cve_id}
                })

        logger.info(f"Processed EPSS data: {len(queries) // 2} entries")
    except Exception as e:
        import traceback
        logger.error(f"Error processing EPSS file {csv_file}: {e}")
        traceback.print_exc()

    return queries


class _DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return float(o)
        return super().default(o)


def _write_nvd_batch(out_dir: Path, idx: int, batch: List[Dict]) -> None:
    out_file = out_dir / f"batch_{idx:05d}.ndjson"
    with out_file.open("w", encoding="utf-8") as f:
        for record in batch:
            f.write(json.dumps(record, ensure_ascii=False, cls=_DecimalEncoder) + "\n")


def download_nvd_data(data_dir: Path) -> None:
    """Download NVD CVE bulk gzip feeds from 2024 onwards.

    Uses the static gzip feed endpoint (no pagination, no rate limiting).
    Output: threat_data/nvd/{year}/batch_NNNNN.ndjson
    Past years are skipped if their directory already contains batch files.
    The current year is always re-downloaded (updated daily by NVD).
    """
    nvd_dir = data_dir / "nvd"
    nvd_dir.mkdir(exist_ok=True)

    current_year = datetime.now().year

    nvd_session = requests.Session()
    nvd_session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    nvd_session.headers.update({
        "User-Agent": "nvd-feed-downloader/1.0",
        "Accept": "*/*",
    })
    if nvd_api_key:
        nvd_session.headers.update({"apiKey": nvd_api_key})

    batch_size = 5000

    for year in range(2024, current_year + 1):
        year_dir = nvd_dir / str(year)
        is_current_year = (year == current_year)

        if year_dir.exists() and any(year_dir.glob("batch_*.ndjson")) and not is_current_year:
            logger.info(f"NVD {year} data already exists - skipping download")
            continue

        year_dir.mkdir(exist_ok=True)
        for old_file in year_dir.glob("batch_*.ndjson"):
            old_file.unlink()

        url = f"{DATA_SOURCES['nvd']}/nvdcve-2.0-{year}.json.gz"
        logger.info(f"Downloading NVD feed: {url}")

        try:
            with nvd_session.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                total = 0
                batch_idx = 1
                batch: List[Dict] = []

                with gzip.GzipFile(fileobj=resp.raw) as gz:
                    for vuln in ijson.items(gz, "vulnerabilities.item"):
                        batch.append(vuln)
                        total += 1
                        if len(batch) >= batch_size:
                            _write_nvd_batch(year_dir, batch_idx, batch)
                            batch_idx += 1
                            batch = []

                if batch:
                    _write_nvd_batch(year_dir, batch_idx, batch)

            logger.info(f"NVD {year} saved to {year_dir} ({total} records, {batch_idx} batch files)")
        except requests.RequestException as e:
            logger.error(f"Failed to download NVD feed for {year}: {e}")




def process_engage_data(engage_dir: Path) -> List[Dict]:
    """Process MITRE Engage data - creates goal, approach, activity, vulnerability,
    attack_technique, attack_tactic, and reference nodes."""
    queries = []
    json_dir = engage_dir / "Data" / "json"

    if not json_dir.exists():
        logger.warning(f"Engage JSON directory not found: {json_dir}")
        return queries

    def load(filename):
        path = json_dir / filename
        if path.exists():
            return load_json_file(str(path))
        logger.warning(f"Engage file not found: {filename}")
        return None

    # --- Goals ---
    goals_data = load("goals.json") or []
    goal_details = load("goal_details.json") or {}

    for goal in goals_data:
        gid = goal.get("id", "")
        if not gid:
            continue
        detail = goal_details.get(gid, {})
        stix_id = generate_stix_id("engage", gid)
        properties = {
            "id": gid,
            "name": goal.get("name", "N/A"),
            "description": goal.get("description", "N/A"),
            "long_description": goal.get("long_description", "N/A"),
            "type": detail.get("type", "N/A"),
        }
        queries.append({
            "statement": create_node_query("goal", stix_id, properties),
            "parameters": {**properties, "stix_id": stix_id}
        })

    # --- Approaches ---
    approaches_data = load("approaches.json") or []
    approach_details = load("approach_details.json") or {}

    for approach in approaches_data:
        aid = approach.get("id", "")
        if not aid:
            continue
        detail = approach_details.get(aid, {})
        stix_id = generate_stix_id("engage", aid)
        properties = {
            "id": aid,
            "name": approach.get("name", "N/A"),
            "description": approach.get("description", "N/A"),
            "long_description": approach.get("long_description", "N/A"),
            "type": detail.get("type", "N/A"),
        }
        queries.append({
            "statement": create_node_query("approach", stix_id, properties),
            "parameters": {**properties, "stix_id": stix_id}
        })

    # --- Activities ---
    activities_data = load("activities.json") or []
    activity_details = load("activity_details.json") or {}

    for activity in activities_data:
        acid = activity.get("id", "")
        if not acid:
            continue
        detail = activity_details.get(acid, {})
        stix_id = generate_stix_id("engage", acid)
        properties = {
            "id": acid,
            "name": activity.get("name", "N/A"),
            "description": activity.get("description", "N/A"),
            "long_description": activity.get("long_description", "N/A"),
            "type": detail.get("type", "N/A"),
        }
        queries.append({
            "statement": create_node_query("activity", stix_id, properties),
            "parameters": {**properties, "stix_id": stix_id}
        })

    # --- Vulnerabilities (from attack_mapping.json: eav_id, eav) ---
    attack_mapping = load("attack_mapping.json") or []
    seen_vulns = set()

    for entry in attack_mapping:
        eav_id = entry.get("eav_id", "")
        if not eav_id or eav_id in seen_vulns:
            continue
        seen_vulns.add(eav_id)
        stix_id = generate_stix_id("engage", eav_id)
        properties = {
            "id": eav_id,
            "eav": entry.get("eav", "N/A"),
        }
        queries.append({
            "statement": create_node_query("vulnerability", stix_id, properties),
            "parameters": {**properties, "stix_id": stix_id}
        })

    # --- Attack Techniques and Tactics (from attack_tactics_techniques.json) ---
    att_data = load("attack_tactics_techniques.json") or {}
    seen_tactics = set()

    for tech_id, entry in att_data.items():
        tech_info = entry.get("attack_technique", {})
        stix_id = generate_stix_id("engage", tech_id)
        properties = {
            "id": tech_id,
            "name": tech_info.get("name", "N/A"),
        }
        queries.append({
            "statement": create_node_query("attack_technique", stix_id, properties),
            "parameters": {**properties, "stix_id": stix_id}
        })

        for tactic in entry.get("attack_tactics", []):
            tac_id = tactic.get("id", "")
            if not tac_id or tac_id in seen_tactics:
                continue
            seen_tactics.add(tac_id)
            tac_stix_id = generate_stix_id("engage", tac_id)
            tac_props = {
                "id": tac_id,
                "name": tactic.get("name", "N/A"),
            }
            queries.append({
                "statement": create_node_query("attack_tactic", tac_stix_id, tac_props),
                "parameters": {**tac_props, "stix_id": tac_stix_id}
            })

    # --- References ---
    references_data = load("references.json") or []
    seen_refs = set()

    for ref in references_data:
        ref_id = ref.get("id", "")
        if not ref_id or ref_id in seen_refs:
            continue
        seen_refs.add(ref_id)
        stix_id = generate_stix_id("engage", ref_id)
        properties = {
            "id": ref_id,
            "title": ref.get("title", "N/A"),
            "url": ref.get("url", "N/A"),
        }
        queries.append({
            "statement": create_node_query("reference", stix_id, properties),
            "parameters": {**properties, "stix_id": stix_id}
        })

    logger.info(f"Generated {len(queries)} Engage node queries "
                f"({len(goals_data)} goals, {len(approaches_data)} approaches, "
                f"{len(activities_data)} activities, {len(seen_vulns)} vulnerabilities, "
                f"{len(att_data)} attack_techniques, {len(seen_tactics)} attack_tactics, "
                f"{len(seen_refs)} references)")
    return queries


def create_engage_relationships(engage_dir: Path) -> List[Dict]:
    """Create all Engage internal relationships and cross-ATT&CK links.

    From activity_details.json:
      activity -[:has_goal]->         goal
      activity -[:has_approach]->     approach
      activity -[:for_vulnerability]-> vulnerability
      activity -[:uses_technique]->   attack_technique
      activity -[:with_tactic]->      attack_tactic
      activity -[:has_reference]->    reference
      attack_tactic -[:maps_ack]->    ATTACK:x-mitre-tactic
      vulnerability -[:with_tactic]-> attack_tactic

    From approach_details.json:
      approach -[:with_goal]->    goal
      approach -[:for_activity]-> activity

    From goal_approach_mappings.json + goal_details.json:
      goal -[:for_approach]-> approach

    From attack_mapping.json:
      attack_technique -[:maps_ap]->        ATTACK:attack-pattern
      attack_technique -[:for_vulnerability]-> vulnerability
      attack_technique -[:for_activity]->   activity

    From attack_tactics_techniques.json:
      attack_technique -[:achieves]-> ATTACK:x-mitre-tactic

    From references.json:
      reference -[:refers]-> activity
    """
    queries = []
    json_dir = engage_dir / "Data" / "json"

    if not json_dir.exists():
        logger.warning(f"Engage JSON directory not found: {json_dir}")
        return queries

    def load(filename):
        path = json_dir / filename
        if path.exists():
            return load_json_file(str(path))
        logger.warning(f"Engage file not found: {filename}")
        return None

    # --- From activity_details.json ---
    activity_details = load("activity_details.json") or {}
    seen_tac_maps = set()

    for acid, detail in activity_details.items():
        acid_stix = generate_stix_id("engage", acid)

        # activity -[:has_goal]-> goal
        for goal_id in detail.get("goals", []):
            if not goal_id:
                continue
            g_stix = generate_stix_id("engage", str(goal_id))
            queries.append({"statement": (
                "MATCH (a:activity {stix_id: $acid}), (g:goal {stix_id: $gid}) "
                "MERGE (a)-[:has_goal]->(g)"
            ), "parameters": {"acid": acid_stix, "gid": g_stix}})

        # activity -[:has_approach]-> approach
        for approach_id in detail.get("approaches", []):
            if not approach_id:
                continue
            ap_stix = generate_stix_id("engage", str(approach_id))
            queries.append({"statement": (
                "MATCH (a:activity {stix_id: $acid}), (ap:approach {stix_id: $apid}) "
                "MERGE (a)-[:has_approach]->(ap)"
            ), "parameters": {"acid": acid_stix, "apid": ap_stix}})

        # activity -[:for_vulnerability]-> vulnerability
        # vulnerabilities is a list of dicts: {"id": "EAV0001", "eav": "..."}
        for vuln_obj in detail.get("vulnerabilities", []):
            eav_id = vuln_obj.get("id", "") if isinstance(vuln_obj, dict) else str(vuln_obj)
            if not eav_id:
                continue
            v_stix = generate_stix_id("engage", eav_id)
            queries.append({"statement": (
                "MATCH (a:activity {stix_id: $acid}), (v:vulnerability {stix_id: $vid}) "
                "MERGE (a)-[:for_vulnerability]->(v)"
            ), "parameters": {"acid": acid_stix, "vid": v_stix}})

        # activity -[:uses_technique]-> attack_technique
        # attack_techniques is a list of dicts: {"id": "T1007", "name": "...", "attack_tactics": [...]}
        for tech_obj in detail.get("attack_techniques", []):
            tech_id = tech_obj.get("id", "") if isinstance(tech_obj, dict) else str(tech_obj)
            if not tech_id:
                continue
            t_stix = generate_stix_id("engage", tech_id)
            queries.append({"statement": (
                "MATCH (a:activity {stix_id: $acid}), (t:attack_technique {stix_id: $tid}) "
                "MERGE (a)-[:uses_technique]->(t)"
            ), "parameters": {"acid": acid_stix, "tid": t_stix}})

        # activity -[:with_tactic]-> attack_tactic
        # attack_tactic -[:maps_ack]-> x-mitre-tactic  (deduplicated)
        # vulnerability -[:with_tactic]-> attack_tactic (cross-product within activity)
        # attack_tactics is a list of dicts: {"id": "TA0007", "name": "Discovery"}
        tac_ids = [t.get("id", "") for t in detail.get("attack_tactics", [])
                   if isinstance(t, dict) and t.get("id")]
        # eav_ids extracted from vulnerability dicts
        eav_ids = [v.get("id", "") for v in detail.get("vulnerabilities", [])
                   if isinstance(v, dict) and v.get("id")]

        for tac_id in tac_ids:
            tac_stix = generate_stix_id("engage", tac_id)
            queries.append({"statement": (
                "MATCH (a:activity {stix_id: $acid}), (tac:attack_tactic {stix_id: $tacid}) "
                "MERGE (a)-[:with_tactic]->(tac)"
            ), "parameters": {"acid": acid_stix, "tacid": tac_stix}})

            # attack_tactic -[:maps_ack]-> x-mitre-tactic (once per tac_id)
            if tac_id not in seen_tac_maps:
                seen_tac_maps.add(tac_id)
                queries.append({"statement": (
                    "MATCH (tac:attack_tactic {stix_id: $tacid}), "
                    "(xt:`x-mitre-tactic` {mitre_id: $mitre_id}) "
                    "MERGE (tac)-[:maps_ack]->(xt)"
                ), "parameters": {"tacid": tac_stix, "mitre_id": tac_id}})

            # vulnerability -[:with_tactic]-> attack_tactic
            for eav_id in eav_ids:
                v_stix = generate_stix_id("engage", eav_id)
                queries.append({"statement": (
                    "MATCH (v:vulnerability {stix_id: $vid}), (tac:attack_tactic {stix_id: $tacid}) "
                    "MERGE (v)-[:with_tactic]->(tac)"
                ), "parameters": {"vid": v_stix, "tacid": tac_stix}})

        # activity -[:has_reference]-> reference
        # references is a list of dicts: {"id": "REF0005", "title": "...", "url": "..."}
        for ref_obj in detail.get("references", []):
            ref_id = ref_obj.get("id", "") if isinstance(ref_obj, dict) else str(ref_obj)
            if not ref_id:
                continue
            r_stix = generate_stix_id("engage", ref_id)
            queries.append({"statement": (
                "MATCH (a:activity {stix_id: $acid}), (r:reference {stix_id: $rid}) "
                "MERGE (a)-[:has_reference]->(r)"
            ), "parameters": {"acid": acid_stix, "rid": r_stix}})

    # --- From approach_details.json ---
    approach_details = load("approach_details.json") or {}

    for apid, detail in approach_details.items():
        ap_stix = generate_stix_id("engage", apid)

        # approach -[:with_goal]-> goal
        for goal_id in detail.get("goals", []):
            if not goal_id:
                continue
            g_stix = generate_stix_id("engage", str(goal_id))
            queries.append({"statement": (
                "MATCH (ap:approach {stix_id: $apid}), (g:goal {stix_id: $gid}) "
                "MERGE (ap)-[:with_goal]->(g)"
            ), "parameters": {"apid": ap_stix, "gid": g_stix}})

        # approach -[:for_activity]-> activity
        for acid in detail.get("activities", []):
            if not acid:
                continue
            acid_stix = generate_stix_id("engage", str(acid))
            queries.append({"statement": (
                "MATCH (ap:approach {stix_id: $apid}), (a:activity {stix_id: $acid}) "
                "MERGE (ap)-[:for_activity]->(a)"
            ), "parameters": {"apid": ap_stix, "acid": acid_stix}})

    # --- goal -[:for_approach]-> approach ---
    # Source 1: goal_approach_mappings.json
    goal_approach_mappings = load("goal_approach_mappings.json") or []
    seen_goal_approach = set()

    for mapping in goal_approach_mappings:
        goal_id = mapping.get("goal_id", "")
        approach_id = mapping.get("approach_id", "")
        if not goal_id or not approach_id:
            continue
        pair = (goal_id, approach_id)
        if pair in seen_goal_approach:
            continue
        seen_goal_approach.add(pair)
        g_stix = generate_stix_id("engage", str(goal_id))
        ap_stix = generate_stix_id("engage", str(approach_id))
        queries.append({"statement": (
            "MATCH (g:goal {stix_id: $gid}), (ap:approach {stix_id: $apid}) "
            "MERGE (g)-[:for_approach]->(ap)"
        ), "parameters": {"gid": g_stix, "apid": ap_stix}})

    # Source 2: goal_details.json
    goal_details = load("goal_details.json") or {}

    for goal_id, detail in goal_details.items():
        g_stix = generate_stix_id("engage", str(goal_id))
        for approach_id in detail.get("approaches", []):
            if not approach_id:
                continue
            pair = (goal_id, approach_id)
            if pair in seen_goal_approach:
                continue
            seen_goal_approach.add(pair)
            ap_stix = generate_stix_id("engage", str(approach_id))
            queries.append({"statement": (
                "MATCH (g:goal {stix_id: $gid}), (ap:approach {stix_id: $apid}) "
                "MERGE (g)-[:for_approach]->(ap)"
            ), "parameters": {"gid": g_stix, "apid": ap_stix}})

    # --- From attack_mapping.json ---
    attack_mapping = load("attack_mapping.json") or []

    for entry in attack_mapping:
        tech_id  = entry.get("attack_id", "")   # T-code
        eav_id   = entry.get("eav_id", "")
        eac_id   = entry.get("eac_id", "")       # activity ID

        if not tech_id:
            continue
        t_stix = generate_stix_id("engage", tech_id)

        # attack_technique -[:maps_ap]-> attack-pattern
        queries.append({"statement": (
            "MATCH (t:attack_technique {stix_id: $tid}), "
            "(ap:`attack-pattern` {mitre_id: $mitre_id}) "
            "MERGE (t)-[:maps_ap]->(ap)"
        ), "parameters": {"tid": t_stix, "mitre_id": tech_id}})

        # attack_technique -[:for_vulnerability]-> vulnerability
        if eav_id:
            v_stix = generate_stix_id("engage", eav_id)
            queries.append({"statement": (
                "MATCH (t:attack_technique {stix_id: $tid}), (v:vulnerability {stix_id: $vid}) "
                "MERGE (t)-[:for_vulnerability]->(v)"
            ), "parameters": {"tid": t_stix, "vid": v_stix}})

        # attack_technique -[:for_activity]-> activity
        if eac_id:
            acid_stix = generate_stix_id("engage", eac_id)
            queries.append({"statement": (
                "MATCH (t:attack_technique {stix_id: $tid}), (a:activity {stix_id: $acid}) "
                "MERGE (t)-[:for_activity]->(a)"
            ), "parameters": {"tid": t_stix, "acid": acid_stix}})

    # --- attack_technique -[:achieves]-> x-mitre-tactic (from attack_tactics_techniques.json) ---
    att_data = load("attack_tactics_techniques.json") or {}

    for tech_id, entry in att_data.items():
        t_stix = generate_stix_id("engage", tech_id)
        for tactic in entry.get("attack_tactics", []):
            tac_id = tactic.get("id", "")
            if not tac_id:
                continue
            queries.append({"statement": (
                "MATCH (t:attack_technique {stix_id: $tid}), "
                "(xt:`x-mitre-tactic` {mitre_id: $mitre_id}) "
                "MERGE (t)-[:achieves]->(xt)"
            ), "parameters": {"tid": t_stix, "mitre_id": tac_id}})

    # --- reference -[:refers]-> activity (from references.json via activity_id) ---
    references_data = load("references.json") or []

    for ref in references_data:
        ref_id = ref.get("id", "")
        activity_id = ref.get("activity_id", "")
        if not ref_id or not activity_id:
            continue
        r_stix = generate_stix_id("engage", ref_id)
        acid_stix = generate_stix_id("engage", activity_id)
        queries.append({"statement": (
            "MATCH (r:reference {stix_id: $rid}), (a:activity {stix_id: $acid}) "
            "MERGE (r)-[:refers]->(a)"
        ), "parameters": {"rid": r_stix, "acid": acid_stix}})

    logger.info(f"Generated {len(queries)} Engage relationship queries")
    return queries


def process_attack_relationships(cti_dir: Path) -> List[Dict]:
    """Process ATT&CK internal relationships from relationship files."""
    queries = []
    relationship_dir = cti_dir / "enterprise-attack" / "relationship"
    
    if not relationship_dir.exists():
        logger.warning(f"ATT&CK relationship directory not found: {relationship_dir}")
        return queries
    
    logger.info(f"Processing ATT&CK relationships from: {relationship_dir}")
    
    # Map STIX types to our node types - using target schema labels
    type_map = {
        'campaign': 'campaign',
        'malware': 'malware',
        'tool': 'tool',
        'attack-pattern': 'attack-pattern',
        'x-mitre-tactic': 'x-mitre-tactic',
        'course-of-action': 'course-of-action',
        'intrusion-set': 'intrusion-set',
        'identity': 'identity',
        'x-mitre-data-source': 'x-mitre-data-source',
        'x-mitre-data-component': 'x-mitre-data-component',
        'x-mitre-detection-strategy': 'x-mitre-detection-strategy',
        'x-mitre-analytic': 'x-mitre-analytic'
    }
    
    # Count processed relationships
    processed_count = 0
    skipped_count = 0
    
    for file_path in relationship_dir.glob('*.json'):
        try:
            data = load_json_file(str(file_path))
            relationship = data['objects'][0]
            
            # Get source and target STIX IDs
            source_ref = relationship.get('source_ref', '')
            target_ref = relationship.get('target_ref', '')
            relationship_type = relationship.get('relationship_type', '')
            
            if not source_ref or not target_ref:
                continue
            
            # Extract STIX types from IDs
            source_type = source_ref.split('--')[0]
            target_type = target_ref.split('--')[0]
            
            # Map to our node types
            source_node_type = type_map.get(source_type)
            target_node_type = type_map.get(target_type)
            
            if not source_node_type or not target_node_type:
                skipped_count += 1
                logger.debug(f"Skipping relationship: {source_type} -> {target_type} (not in schema)")
                continue
            
            # Use the STIX relationship type directly (keep hyphenated names)
            rel_type = relationship_type if relationship_type else 'RELATED_TO'
            
            # Prepare relationship properties
            rel_props = {
                'created': to_neo4j_datetime(relationship.get('created')),
                'modified': to_neo4j_datetime(relationship.get('modified')),
                'description': relationship.get('description', 'N/A'),
                'relationship_type': relationship_type or 'N/A',
                'x_mitre_deprecated': relationship.get('x_mitre_deprecated', False)
            }
            
            # Use backticks for labels/relationships with hyphens
            source_label = f"`{source_node_type}`" if '-' in source_node_type else source_node_type
            target_label = f"`{target_node_type}`" if '-' in target_node_type else target_node_type
            rel_label = f"`{rel_type}`" if '-' in rel_type else rel_type
            
            # Create relationship query with parameters (safer than string interpolation)
            # Use datetime() for created/modified if not "N/A"
            created_expr = "datetime($created)" if rel_props['created'] != "N/A" else "$created"
            modified_expr = "datetime($modified)" if rel_props['modified'] != "N/A" else "$modified"
            
            query = f"""
            MATCH (a:{source_label} {{stix_id: $source_ref}})
            MATCH (b:{target_label} {{stix_id: $target_ref}})
            MERGE (a)-[r:{rel_label}]->(b)
            SET r.created = {created_expr},
                r.modified = {modified_expr},
                r.description = $description,
                r.relationship_type = $relationship_type,
                r.x_mitre_deprecated = $x_mitre_deprecated
            """
            
            # Create parameters dict
            parameters = {
                'source_ref': source_ref,
                'target_ref': target_ref,
                'created': rel_props['created'],
                'modified': rel_props['modified'],
                'description': rel_props['description'],
                'relationship_type': rel_props['relationship_type'],
                'x_mitre_deprecated': rel_props['x_mitre_deprecated']
            }
            
            queries.append({'statement': query, 'parameters': parameters})
            processed_count += 1
                
        except Exception as e:
            logger.warning(f"Error processing relationship file {file_path}: {e}")
            continue
    
    logger.info(f"Processed {processed_count} ATT&CK relationships, skipped {skipped_count} (unsupported node types)")
    return queries


def create_capec_external_relationships(xml_file: Path) -> List[Dict]:
    """Create CAPEC external relationships: CAPEC -[:map_ap]-> attack-pattern and CAPEC -[:exploits]-> CWE."""
    queries = []

    if not xml_file.exists():
        logger.warning(f"CAPEC XML not found: {xml_file}")
        return queries

    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'capec': 'http://capec.mitre.org/capec-3'}

    map_ap_count = 0
    exploits_count = 0

    for pattern in root.findall('.//capec:Attack_Pattern', ns):
        capec_id = pattern.get('ID')
        if not capec_id:
            continue
        capec_stix_id = generate_stix_id('capec', f"CAPEC-{capec_id}")

        # CAPEC -[:map_ap]-> attack-pattern (via Taxonomy_Mappings for ATT&CK)
        for mapping in pattern.findall('.//capec:Taxonomy_Mapping', ns):
            taxonomy_name = mapping.get('Taxonomy_Name', '')
            if 'ATT&CK' in taxonomy_name or 'ATTACK' in taxonomy_name:
                entry_id_elem = mapping.find('capec:Entry_ID', ns)
                if entry_id_elem is not None and entry_id_elem.text:
                    entry_id = entry_id_elem.text.strip()
                    attack_id = f"T{entry_id}" if not entry_id.startswith('T') else entry_id
                    query = f"""
                    MATCH (t:`attack-pattern`) WHERE t.mitre_id = '{attack_id}'
                    MATCH (c:CAPEC {{stix_id: '{capec_stix_id}'}})
                    MERGE (c)-[:map_ap]->(t)
                    """
                    queries.append({'statement': query})
                    map_ap_count += 1

        # CAPEC -[:exploits]-> CWE (via Related_Weaknesses)
        for weakness in pattern.findall('.//capec:Related_Weakness', ns):
            cwe_id = weakness.get('CWE_ID')
            if cwe_id:
                queries.append({
                    'statement': (
                        "MATCH (c:CAPEC {stix_id: $capec_stix_id}) "
                        "MATCH (w:Weakness {id: $weakness_id}) "
                        "MERGE (c)-[:exploits]->(w)"
                    ),
                    'parameters': {
                        'capec_stix_id': capec_stix_id,
                        'weakness_id': f"CWE-{cwe_id}"
                    }
                })
                exploits_count += 1

    logger.info(f"Created {map_ap_count} CAPEC -[:map_ap]-> attack-pattern relationships")
    logger.info(f"Created {exploits_count} CAPEC -[:exploits]-> CWE relationships")
    return queries


def create_cross_framework_relationships(data_dir: Path) -> List[Dict]:
    """Create relationships between different frameworks."""
    queries = []
    
    # CWE -> CVE relationships
    print("Creating CWE -> CVE relationships...")
    cwe_cve_count = 0
    cve_dir = data_dir / "cve"
    if cve_dir.exists():
        all_cve_files = list(cve_dir.rglob("*.json"))
        filtered_cve_files = filter_cve_files(all_cve_files)
        
        for json_file in filtered_cve_files:
            try:
                data = load_json_file(str(json_file))
                
                if data.get('dataType') == 'CVE_RECORD':
                    cve_id = data.get('cveMetadata', {}).get('cveId')
                    if cve_id:
                        # Look for CWE references in problem types
                        containers = data.get('containers', {})
                        
                        # Handle containers as either dict or list
                        if isinstance(containers, dict):
                            containers = [containers]
                        elif not isinstance(containers, list):
                            containers = []
                        
                        for container in containers:
                            cna = container.get('cna', {})
                            problem_types = cna.get('problemTypes', [])
                            if isinstance(problem_types, list):
                                for problem_type in problem_types:
                                    if isinstance(problem_type, dict):
                                        descriptions = problem_type.get('descriptions', [])
                                        if isinstance(descriptions, list):
                                            for desc in descriptions:
                                                if isinstance(desc, dict):
                                                    # Check for cweId field first (preferred method)
                                                    cwe_id = desc.get('cweId')
                                                    if cwe_id:
                                                        # Extract just the number from CWE-XXX format
                                                        cwe_match = re.search(r'CWE-(\d+)', cwe_id)
                                                        if cwe_match:
                                                            cwe_id_num = cwe_match.group(1)
                                                            cwe_weakness_id = f"CWE-{cwe_id_num}"
                                                            
                                                            query = f"""
                                                            MATCH (w:Weakness {{id: '{cwe_weakness_id}'}})
                                                            MATCH (v:CVE {{id: '{cve_id}'}})
                                                            MERGE (w)-[:observed_as]->(v)
                                                            """
                                                            queries.append({'statement': query})
                                                            cwe_cve_count += 1
                                                    else:
                                                        # Fallback: try to extract from description text
                                                        cwe_match = re.search(r'CWE-(\d+)', desc.get('description', ''))
                                                        if cwe_match:
                                                            cwe_id_num = cwe_match.group(1)
                                                            cwe_weakness_id = f"CWE-{cwe_id_num}"
                                                            
                                                            query = f"""
                                                            MATCH (w:Weakness {{id: '{cwe_weakness_id}'}})
                                                            MATCH (v:CVE {{id: '{cve_id}'}})
                                                            MERGE (w)-[:observed_as]->(v)
                                                            """
                                                            queries.append({'statement': query})
                                                            cwe_cve_count += 1
            except Exception as e:
                continue
    
    print(f"Created {cwe_cve_count} CWE -> CVE relationships")
    # Note: Weakness -[:known_exploit]-> KEV is handled in process_kev_data()

    # Note: CVE -[:impacts]-> CAPEC is handled directly in process_cve_data()
    # from containers/cna/impacts/[]/capecId

    print(f"Total cross-framework relationships: {len(queries)}")
    
    return queries



def execute_queries(queries: List[Dict], batch_size: int = 1000):
    """Execute a batch of queries against Neo4j with retry logic.

    Performance notes:
    - Bolt path: uses a module-level driver singleton (no reconnect per call)
      and runs each batch inside a single explicit write transaction, which
      is dramatically faster than one auto-commit transaction per query.
    - HTTP path: unchanged (sends batch as a single /tx/commit payload).
    """
    if not queries:
        return

    if use_bolt:
        driver = _get_bolt_driver()

        def _run_batch(tx, batch):
            for query_item in batch:
                tx.run(query_item.get('statement', ''), query_item.get('parameters', {}))

        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_num = i // batch_size + 1

            if i == 0:
                logger.info("Executing Neo4j queries via Bolt protocol")

            max_retries = 3
            retry_delay = 1

            for attempt in range(max_retries):
                try:
                    with driver.session(database=neo4j_db) as db_session:
                        db_session.execute_write(_run_batch, batch)

                    if batch_num % 10 == 0 or i + batch_size >= len(queries):
                        logger.info(f"Successfully executed batch {batch_num}")
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to execute batch after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
    else:
        # Use HTTP REST API for http:// URIs
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            payload = {'statements': batch}
            max_retries = 3
            retry_delay = 1

            if i == 0:
                logger.info(f"Executing Neo4j queries against: {NEO4J_ENDPOINT}")

            for attempt in range(max_retries):
                try:
                    response = session.post(NEO4J_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
                    response.raise_for_status()

                    result = response.json()
                    if 'errors' in result and result['errors']:
                        logger.error(f"Neo4j returned errors: {result['errors']}")
                    else:
                        logger.info(f"Successfully executed batch {i//batch_size + 1}")
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to execute batch after {max_retries} attempts: {str(e)}")
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

def create_custom_attack_relationships(cti_dir: Path) -> List[Dict]:
    """Create custom ATT&CK relationships derived from node properties (not STIX relationship files).

    Implements:
      - (x-mitre-analytic)-[:requires_data]->(x-mitre-data-component)
          Source: x_mitre_log_source_references[].x_mitre_data_component_ref on analytic objects
      - (x-mitre-detection-strategy)-[:implemented_by]->(x-mitre-analytic)
          Source: x_mitre_analytic_refs[] on detection-strategy objects
    """
    queries = []
    enterprise_dir = cti_dir / "enterprise-attack"

    # --- (x-mitre-analytic)-[:requires_data]->(x-mitre-data-component) ---
    analytic_dir = enterprise_dir / "x-mitre-analytic"
    if analytic_dir.exists():
        for file_path in analytic_dir.glob('*.json'):
            try:
                data = load_json_file(str(file_path))
                analytic = data['objects'][0]
                if analytic.get('type') != 'x-mitre-analytic':
                    continue
                analytic_id = analytic.get('id', '')
                log_source_refs = analytic.get('x_mitre_log_source_references', [])
                for entry in log_source_refs:
                    dc_ref = entry.get('x_mitre_data_component_ref', '')
                    if analytic_id and dc_ref:
                        query = {
                            'statement': (
                                "MATCH (an:`x-mitre-analytic` {id: $analytic_id}), "
                                "(dc:`x-mitre-data-component` {id: $dc_id}) "
                                "MERGE (an)-[:requires_data]->(dc)"
                            ),
                            'parameters': {'analytic_id': analytic_id, 'dc_id': dc_ref}
                        }
                        queries.append(query)
            except Exception as e:
                logger.warning(f"Error processing analytic requires_data for {file_path}: {e}")

    # --- (x-mitre-detection-strategy)-[:implemented_by]->(x-mitre-analytic) ---
    det_dir = enterprise_dir / "x-mitre-detection-strategy"
    if det_dir.exists():
        for file_path in det_dir.glob('*.json'):
            try:
                data = load_json_file(str(file_path))
                ds = data['objects'][0]
                if ds.get('type') != 'x-mitre-detection-strategy':
                    continue
                ds_id = ds.get('id', '')
                analytic_refs = ds.get('x_mitre_analytic_refs', [])
                for analytic_ref in analytic_refs:
                    if ds_id and analytic_ref:
                        query = {
                            'statement': (
                                "MATCH (det:`x-mitre-detection-strategy` {id: $det_id}), "
                                "(an:`x-mitre-analytic` {id: $analytic_id}) "
                                "MERGE (det)-[:implemented_by]->(an)"
                            ),
                            'parameters': {'det_id': ds_id, 'analytic_id': analytic_ref}
                        }
                        queries.append(query)
            except Exception as e:
                logger.warning(f"Error processing detection-strategy implemented_by for {file_path}: {e}")

    logger.info(f"Generated {len(queries)} custom ATT&CK relationship queries "
                f"(requires_data, implemented_by)")
    return queries

def process_sigma_data(sigma_dir: Path) -> List[Dict]:
    """Process Sigma detection rules — creates SigmaRule nodes and DETECTS relationships to ATT&CK techniques.

    Each YAML rule's `tags` list is scanned for entries matching `attack.tXXXX` / `attack.tXXXX.YYY`
    which are normalised to uppercase ATT&CK IDs (e.g. T1059, T1059.001) and linked to existing
    `attack-pattern` nodes via a [:detects] relationship (same type as ATT&CK data-component detects).
    """
    queries: List[Dict] = []
    rules_dir = sigma_dir / "rules"

    if not rules_dir.exists():
        logger.warning(f"Sigma rules directory not found: {rules_dir}")
        return queries

    rule_count = 0
    rel_count = 0
    error_count = 0

    for yml_file in rules_dir.rglob("*.yml"):
        try:
            with yml_file.open("r", encoding="utf-8") as f:
                rule = yaml.safe_load(f)

            if not isinstance(rule, dict):
                continue

            rule_id = rule.get("id", "")
            if not rule_id:
                continue

            title = rule.get("title", "N/A")
            status = rule.get("status", "N/A")
            level = rule.get("level", "N/A")
            description = rule.get("description", "N/A") or "N/A"
            author = rule.get("author", "N/A") or "N/A"

            logsource = rule.get("logsource", {}) or {}
            logsource_product = logsource.get("product", "N/A") or "N/A"
            logsource_category = logsource.get("category", "N/A") or "N/A"
            logsource_service = logsource.get("service", "N/A") or "N/A"

            detection = to_json_string(rule.get("detection", {}))
            falsepositives = rule.get("falsepositives", []) or []
            if isinstance(falsepositives, str):
                falsepositives = [falsepositives]

            tags = rule.get("tags", []) or []
            if isinstance(tags, str):
                tags = [tags]

            # Extract ATT&CK technique IDs from tags (e.g. attack.t1059.001 → T1059.001)
            technique_ids = []
            for tag in tags:
                tag_lower = str(tag).lower()
                if tag_lower.startswith("attack.t"):
                    raw = tag_lower[len("attack."):]          # e.g. "t1059.001"
                    parts = raw.split(".")
                    if len(parts) == 1:
                        tid = parts[0].upper()               # "T1059"
                    else:
                        tid = f"{parts[0].upper()}.{parts[1].zfill(3)}"  # "T1059.001"
                    technique_ids.append(tid)

            stix_id = generate_stix_id("sigma", rule_id)
            properties = {
                "id": rule_id,
                "title": title,
                "status": status,
                "level": level,
                "description": str(description)[:2000],
                "author": str(author)[:500],
                "logsource_product": logsource_product,
                "logsource_category": logsource_category,
                "logsource_service": logsource_service,
                "detection": detection,
                "falsepositives": falsepositives,
                "tags": [str(t) for t in tags],
            }
            queries.append({
                "statement": create_node_query("SigmaRule", stix_id, properties),
                "parameters": {**properties, "stix_id": stix_id},
            })
            rule_count += 1

            # SigmaRule -[:detects]-> attack-pattern (same type as ATT&CK data-component detects)
            for tid in technique_ids:
                queries.append({
                    "statement": (
                        "MATCH (s:SigmaRule {stix_id: $stix_id}) "
                        "MATCH (ap:`attack-pattern` {mitre_id: $mitre_id}) "
                        "MERGE (s)-[:detects]->(ap)"
                    ),
                    "parameters": {"stix_id": stix_id, "mitre_id": tid},
                })
                rel_count += 1

        except Exception as e:
            logger.warning(f"Error processing Sigma rule {yml_file}: {e}")
            error_count += 1
            continue

    logger.info(
        f"Processed {rule_count} Sigma rules, {rel_count} detects relationships "
        f"({error_count} files skipped due to errors)"
    )
    return queries


def process_exploitdb_data(exploitdb_dir: Path) -> List[Dict]:
    """Process ExploitDB files_exploits.csv — creates ExploitDBEntry nodes linked to CVE nodes."""
    queries: List[Dict] = []
    csv_file = exploitdb_dir / "files_exploits.csv"

    if not csv_file.exists():
        logger.warning(f"ExploitDB CSV not found: {csv_file}")
        return queries

    node_count = 0
    rel_count = 0
    skipped = 0

    try:
        with open(csv_file, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_cves = row.get("codes", "").strip()
                if not raw_cves:
                    skipped += 1
                    continue

                # Normalize multi-CVE field — values may be "CVE-X;CVE-Y" or "CVE-X,CVE-Y"
                cve_ids = [
                    c.strip()
                    for c in re.split(r"[;,]", raw_cves)
                    if c.strip().upper().startswith("CVE-")
                ]
                if not cve_ids:
                    skipped += 1
                    continue

                # Only process CVE-2024+
                cve_ids = [
                    c for c in cve_ids
                    if len(c.split("-")) > 1 and c.split("-")[1].isdigit() and int(c.split("-")[1]) >= 2024
                ]
                if not cve_ids:
                    skipped += 1
                    continue

                edb_id = row.get("id", "").strip()
                if not edb_id:
                    skipped += 1
                    continue

                node_id = f"EDB-{edb_id}"
                stix_id = generate_stix_id("exploitdbentry", node_id)
                published = to_neo4j_datetime(row.get("date_published", "").strip() or None)
                properties = {
                    "id": node_id,
                    "url": f"https://www.exploit-db.com/exploits/{edb_id}",
                    "title": row.get("description", "N/A").strip()[:500],
                    "type": row.get("type", "N/A").strip(),
                    "platform": row.get("platform", "N/A").strip(),
                    "published": published,
                }
                queries.append({
                    "statement": create_node_query("ExploitDBEntry", stix_id, properties),
                    "parameters": {**properties, "stix_id": stix_id},
                })
                node_count += 1

                for cve_id in cve_ids:
                    queries.append({
                        "statement": (
                            "MATCH (e:ExploitDBEntry {stix_id: $stix_id}) "
                            "MATCH (c:CVE {id: $cve_id}) "
                            "MERGE (c)-[:has_weaponized_exploit]->(e)"
                        ),
                        "parameters": {"stix_id": stix_id, "cve_id": cve_id},
                    })
                    rel_count += 1

    except Exception as ex:
        logger.error(f"Error processing ExploitDB CSV: {ex}")

    logger.info(
        f"ExploitDB: {node_count} ExploitDBEntry nodes, {rel_count} has_weaponized_exploit relationships "
        f"({skipped} rows skipped)"
    )
    return queries


def process_poc_github_data(poc_dir: Path) -> List[Dict]:
    """Process poc-in-github JSON files — creates GithubPoC nodes linked to CVE nodes."""
    queries: List[Dict] = []

    if not poc_dir.exists():
        logger.warning(f"poc-in-github directory not found: {poc_dir}")
        return queries

    node_count = 0
    rel_count = 0
    error_count = 0

    for json_file in sorted(poc_dir.rglob("CVE-*.json")):
        try:
            cve_id = json_file.stem  # e.g. "CVE-2024-12345"
            parts = cve_id.split("-")
            if len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) < 2024:
                continue

            repos = load_json_file(str(json_file))
            if not isinstance(repos, list):
                continue

            for repo in repos:
                html_url = repo.get("html_url", "").strip()
                if not html_url:
                    continue

                full_name = repo.get("full_name", html_url).strip()
                node_id = f"POC-{full_name.replace('/', '-')}"
                stix_id = generate_stix_id("githubpoc", node_id)
                published = to_neo4j_datetime(repo.get("created_at", "").strip() or None)
                properties = {
                    "id": node_id,
                    "url": html_url,
                    "title": (repo.get("description") or "N/A")[:500],
                    "published": published,
                }
                queries.append({
                    "statement": create_node_query("GithubPoC", stix_id, properties),
                    "parameters": {**properties, "stix_id": stix_id},
                })
                node_count += 1

                queries.append({
                    "statement": (
                        "MATCH (e:GithubPoC {stix_id: $stix_id}) "
                        "MATCH (c:CVE {id: $cve_id}) "
                        "MERGE (c)-[:has_poc]->(e)"
                    ),
                    "parameters": {"stix_id": stix_id, "cve_id": cve_id},
                })
                rel_count += 1

        except Exception as ex:
            logger.warning(f"Error processing poc-in-github file {json_file}: {ex}")
            error_count += 1
            continue

    logger.info(
        f"poc-in-github: {node_count} GithubPoC nodes, {rel_count} has_poc relationships "
        f"({error_count} files skipped)"
    )
    return queries


def create_constraints():
    """Create Neo4j constraints for better performance - TARGET SCHEMA."""
    constraints = [
        # ATT&CK/STIX nodes - use backticks for hyphenated labels
        "CREATE CONSTRAINT stix_id_attack_pattern IF NOT EXISTS FOR (n:`attack-pattern`) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_tactic IF NOT EXISTS FOR (n:`x-mitre-tactic`) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_campaign IF NOT EXISTS FOR (n:`campaign`) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_course_of_action IF NOT EXISTS FOR (n:`course-of-action`) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_intrusion_set IF NOT EXISTS FOR (n:`intrusion-set`) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_malware IF NOT EXISTS FOR (n:`malware`) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_tool IF NOT EXISTS FOR (n:`tool`) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_data_component IF NOT EXISTS FOR (n:`x-mitre-data-component`) REQUIRE n.stix_id IS UNIQUE",
        
        # CWE nodes - 'id' is the MERGE key; stix_id added for consistency with other data sources
        # Note: Observed_Example is NOT a node - it's a relationship (Weakness)-[:observed_as]->(CVE)
        "CREATE CONSTRAINT weakness_id IF NOT EXISTS FOR (n:Weakness) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_weakness IF NOT EXISTS FOR (n:Weakness) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT observed_example_id IF NOT EXISTS FOR (n:Observed_Example) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_observed_example IF NOT EXISTS FOR (n:Observed_Example) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT mitigation_id IF NOT EXISTS FOR (n:Mitigation) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_mitigation IF NOT EXISTS FOR (n:Mitigation) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT detection_method_id IF NOT EXISTS FOR (n:Detection_Method) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_detection_method IF NOT EXISTS FOR (n:Detection_Method) REQUIRE n.stix_id IS UNIQUE",
        
        # CVE node
        "CREATE CONSTRAINT cve_id IF NOT EXISTS FOR (n:CVE) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_cve IF NOT EXISTS FOR (n:CVE) REQUIRE n.stix_id IS UNIQUE",
        
        # CAPEC node
        "CREATE CONSTRAINT stix_id_capec IF NOT EXISTS FOR (n:CAPEC) REQUIRE n.stix_id IS UNIQUE",
        
        # KEV node
        "CREATE CONSTRAINT stix_id_kev IF NOT EXISTS FOR (n:KEV) REQUIRE n.stix_id IS UNIQUE",

        # EPSS node
        "CREATE CONSTRAINT stix_id_epss IF NOT EXISTS FOR (n:EPSS) REQUIRE n.stix_id IS UNIQUE",

        # Engage nodes
        "CREATE CONSTRAINT stix_id_engage_goal IF NOT EXISTS FOR (n:goal) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_engage_approach IF NOT EXISTS FOR (n:approach) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_engage_activity IF NOT EXISTS FOR (n:activity) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_engage_vulnerability IF NOT EXISTS FOR (n:vulnerability) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_engage_attack_technique IF NOT EXISTS FOR (n:attack_technique) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_engage_attack_tactic IF NOT EXISTS FOR (n:attack_tactic) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT stix_id_engage_reference IF NOT EXISTS FOR (n:reference) REQUIRE n.stix_id IS UNIQUE",

        # Sigma rules
        "CREATE CONSTRAINT stix_id_sigma_rule IF NOT EXISTS FOR (n:SigmaRule) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT sigma_rule_id IF NOT EXISTS FOR (n:SigmaRule) REQUIRE n.id IS UNIQUE",

        # ExploitDBEntry node (curated exploits from Exploit-DB)
        "CREATE CONSTRAINT stix_id_exploitdb_entry IF NOT EXISTS FOR (n:ExploitDBEntry) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT exploitdb_entry_id IF NOT EXISTS FOR (n:ExploitDBEntry) REQUIRE n.id IS UNIQUE",

        # GithubPoC node (community PoC repos from nomi-sec/PoC-in-GitHub)
        "CREATE CONSTRAINT stix_id_github_poc IF NOT EXISTS FOR (n:GithubPoC) REQUIRE n.stix_id IS UNIQUE",
        "CREATE CONSTRAINT github_poc_id IF NOT EXISTS FOR (n:GithubPoC) REQUIRE n.id IS UNIQUE"
    ]
    
    logger.info("Creating constraints...")
    for constraint in constraints:
        try:
            execute_queries([{'statement': constraint}])
        except Exception as e:
            logger.warning(f"Constraint creation warning (may already exist): {e}")

def create_achieves_relationships(cti_dir: Path):
    """Create derived achieves relationships: (attack-pattern)-[:achieves]->(x-mitre-tactic).

    Parses kill_chain_phases from source JSON files and matches phase_name exactly
    to x-mitre-tactic.x_mitre_shortname for kill_chain_name='mitre-attack'.
    """
    logger.info("Creating derived achieves relationships...")

    enterprise_dir = cti_dir / "enterprise-attack" / "attack-pattern"
    if not enterprise_dir.exists():
        logger.warning(f"ATT&CK attack-pattern directory not found: {enterprise_dir}")
        return

    queries = []
    for file_path in enterprise_dir.glob("*.json"):
        try:
            data = load_json_file(str(file_path))
            technique = data['objects'][0]
            if technique.get('type') != 'attack-pattern':
                continue
            stix_id = technique.get('id', '')
            kill_chain_phases = technique.get('kill_chain_phases', [])
            for phase in kill_chain_phases:
                if phase.get('kill_chain_name') != 'mitre-attack':
                    continue
                phase_name = phase.get('phase_name', '')
                if not phase_name:
                    continue
                queries.append({
                    'statement': (
                        "MATCH (ap:`attack-pattern` {stix_id: $stix_id}) "
                        "MATCH (tactic:`x-mitre-tactic` {x_mitre_shortname: $phase_name}) "
                        "MERGE (ap)-[:achieves]->(tactic)"
                    ),
                    'parameters': {'stix_id': stix_id, 'phase_name': phase_name}
                })
        except Exception as e:
            logger.warning(f"Error processing achieves for {file_path}: {e}")
            continue

    try:
        execute_queries(queries)
        logger.info(f"Achieves relationships created successfully ({len(queries)} queries)")
    except Exception as e:
        logger.error(f"Failed to create achieves relationships: {e}")


def create_mitigated_by_relationships():
    """Derive (attack-pattern)-[:mitigated_by]->(course-of-action) as the inverse
    of the already-imported (course-of-action)-[:mitigates]->(attack-pattern).
    Must be called after process_attack_relationships() has run.
    """
    logger.info("Creating derived mitigated_by relationships...")
    query = (
        "MATCH (coa:`course-of-action`)-[:mitigates]->(ap:`attack-pattern`) "
        "MERGE (ap)-[:mitigated_by]->(coa)"
    )
    try:
        execute_queries([{'statement': query}])
        logger.info("mitigated_by relationships created successfully")
    except Exception as e:
        logger.error(f"Failed to create mitigated_by relationships: {e}")


def import_all_data(data_dir: Optional[Path] = None):
    """Main function to import all threat intelligence data."""
    logger.info("Starting complete threat intelligence data import...")
    
    # Use provided data directory or download data
    if data_dir is None:
        data_dir = download_and_extract_data()
    else:
        logger.info(f"Using provided data directory: {data_dir}")
    
    # 1. Create constraints
    logger.info("Creating Neo4j constraints...")
    create_constraints()

    # 2. ATT&CK nodes
    logger.info("Processing ATT&CK data...")
    attack_queries = process_attack_data(data_dir / "cti")
    execute_queries(attack_queries)

    # 3. ATT&CK explicit relationships
    logger.info("Creating ATT&CK internal relationships...")
    attack_relationship_queries = process_attack_relationships(data_dir / "cti")
    execute_queries(attack_relationship_queries)

    # 4. ATT&CK custom relationships
    logger.info("Creating custom ATT&CK relationships...")
    custom_rel_queries = create_custom_attack_relationships(data_dir / "cti")
    execute_queries(custom_rel_queries)

    # 5. ATT&CK achieves relationships
    logger.info("Creating derived achieves relationships...")
    create_achieves_relationships(data_dir / "cti")

    # 5b. Derived mitigated_by (inverse of mitigates)
    create_mitigated_by_relationships()

    # 6. CAPEC nodes
    logger.info("Processing CAPEC data...")
    capec_queries = process_capec_data(data_dir / "capec_latest.xml")
    execute_queries(capec_queries)

    # 7. CAPEC internal relationships (ChildOf, CanPrecede, PeerOf)
    logger.info("Creating CAPEC internal relationships...")
    capec_internal_queries = create_capec_internal_relationships(data_dir / "capec_latest.xml")
    execute_queries(capec_internal_queries)

    # 8. CWE nodes + edges (must be before CAPEC external and KEV)
    cwe_file = next((data_dir / "cwe").glob("*.xml"), None)
    if cwe_file:
        logger.info("Processing CWE data...")
        cwe_queries = process_cwe_data(cwe_file)
        execute_queries(cwe_queries)

    # 9. CAPEC external relationships (exploits -> CWE must exist, map_ap -> ATT&CK must exist)
    logger.info("Creating CAPEC external relationships...")
    capec_external_queries = create_capec_external_relationships(data_dir / "capec_latest.xml")
    execute_queries(capec_external_queries)

    # 10. CVE nodes
    logger.info("Processing CVE data...")
    cve_queries = process_cve_data(data_dir / "cve")
    execute_queries(cve_queries)

    # 11. KEV nodes + known_exploit edges (CWE must exist)
    kev_file = data_dir / "known_exploited_vulnerabilities.json"
    if kev_file.exists():
        logger.info("Processing KEV data...")
        kev_queries = process_kev_data(kev_file)
        execute_queries(kev_queries)

    # 12. Cross-framework relationships (CWE->CVE, others active)
    logger.info("Creating cross-framework relationships...")
    relationship_queries = create_cross_framework_relationships(data_dir)
    execute_queries(relationship_queries)

    # 13. Engage nodes
    logger.info("Processing Engage data...")
    engage_queries = process_engage_data(data_dir / "engage")
    execute_queries(engage_queries)

    # 14. Engage relationships (internal + cross-ATT&CK, ATT&CK must exist)
    logger.info("Creating Engage relationships...")
    engage_rel_queries = create_engage_relationships(data_dir / "engage")
    execute_queries(engage_rel_queries)

    # 15. EPSS nodes + scores edges — always re-download and update (CVE must exist)
    logger.info("Downloading and updating EPSS data...")
    epss_file = download_epss_data(data_dir)
    if epss_file.exists():
        epss_queries = process_epss_data(epss_file)
        execute_queries(epss_queries)
    else:
        logger.warning("EPSS CSV not found after download, skipping EPSS processing")

    # 16. Sigma rules nodes + detects edges (ATT&CK attack-pattern nodes must exist)
    logger.info("Processing Sigma detection rules...")
    sigma_queries = process_sigma_data(data_dir / "sigma")
    execute_queries(sigma_queries)

    # 17. ExploitDBEntry + GithubPoC nodes (CVE nodes must exist)
    logger.info("Processing ExploitDB entries...")
    exploitdb_queries = process_exploitdb_data(data_dir / "exploitdb")
    execute_queries(exploitdb_queries)

    logger.info("Processing poc-in-github PoCs...")
    poc_queries = process_poc_github_data(data_dir / "poc_github")
    execute_queries(poc_queries)

    logger.info("="*60)
    logger.info("Import completed successfully!")
    logger.info("="*60)

if __name__ == "__main__":
    import_all_data()