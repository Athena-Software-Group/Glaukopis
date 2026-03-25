# Athena Threat Intelligence Database - Local Setup Guide

This guide provides step-by-step instructions for setting up and running the Athena Threat Intelligence Database on your local machine without Docker.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Step 1: Install Neo4j Desktop](#step-1-install-neo4j-desktop)
- [Step 2: Create a New Database](#step-2-create-a-new-database)
- [Step 3: Install APOC Plugin](#step-3-install-apoc-plugin)
- [Step 4: Configure Database Settings](#step-4-configure-database-settings)
- [Step 5: Start the Database](#step-5-start-the-database)
- [Step 6: Set Up Python Environment](#step-6-set-up-python-environment)
- [Step 7: Configure Database Credentials](#step-7-configure-database-credentials)
- [Step 8: Run the Population Script](#step-8-run-the-population-script)
- [Step 9: Verify the Installation](#step-9-verify-the-installation)
- [Data Sources](#data-sources)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for large datasets)
- **Disk Space**: At least 20GB free space
- **Internet Connection**: Required for downloading threat intelligence data
- **Git**: Required for cloning MITRE CTI and CVE repositories

---

## Step 1: Install Neo4j Desktop

### Download Neo4j Desktop

1. Visit the official Neo4j Desktop download page: [https://neo4j.com/download/](https://neo4j.com/download/)
2. Click on **"Download Desktop"**
3. Fill in the required information (name and email) to receive an activation key
4. Download the installer for your operating system:
   - **Windows**: `Neo4j Desktop Setup.exe`
   - **macOS**: `Neo4j Desktop.dmg`
   - **Linux**: `Neo4j Desktop.AppImage`

### Install Neo4j Desktop

#### Windows
1. Run the downloaded `.exe` file
2. Follow the installation wizard
3. Accept the license agreement
4. Choose installation directory
5. Complete the installation

#### macOS
1. Open the downloaded `.dmg` file
2. Drag Neo4j Desktop to your Applications folder
3. Open Neo4j Desktop from Applications
4. If prompted, allow the app in System Preferences → Security & Privacy

#### Linux
1. Make the AppImage executable:
   ```bash
   chmod +x Neo4j-Desktop-*.AppImage
   ```
2. Run the AppImage:
   ```bash
   ./Neo4j-Desktop-*.AppImage
   ```

### Activate Neo4j Desktop

1. Launch Neo4j Desktop
2. Enter the activation key received via email
3. Click **"Activate"**

---

## Step 2: Create a New Database

1. **Open Neo4j Desktop**
2. **Create a New Project**:
   - Click on **"New"** or **"+ New Project"** in the left sidebar
   - Name your project (e.g., "Athena Threat Intelligence")

3. **Add a Database**:
   - Inside your project, click **"Add"** → **"Local DBMS"**
   - Configure the database:
     - **Name**: `athena-threat-db`
     - **Password**: Choose a strong password (e.g., `Graph@123`)
     - **Version**: Select Neo4j 5.x (latest stable version)
   - Click **"Create"**

---

## Step 3: Install APOC Plugin

The APOC (Awesome Procedures on Cypher) plugin is required for advanced graph operations.

### Install APOC via Neo4j Desktop

1. **Select Your Database**:
   - In Neo4j Desktop, click on your database (`athena-threat-db`)

2. **Open Plugins Tab**:
   - Click on the **"Plugins"** tab (icon looks like a puzzle piece)

3. **Install APOC**:
   - Find **"APOC"** in the list of available plugins
   - Click **"Install"** next to APOC
   - Wait for the installation to complete (you'll see a green checkmark)

4. **Verify Installation**:
   - The APOC plugin should now show as "Installed"

> **Note**: If you're using Neo4j 5.x, APOC Core is included by default. You may need to install APOC Extended for additional procedures.

---

## Step 4: Configure Database Settings

1. In Neo4j Desktop, click the **"..."** menu next to your database and select **"Settings"**
2. Add or verify the following configuration:

```properties
# Enable APOC procedures
dbms.security.procedures.unrestricted=apoc.*
dbms.security.procedures.allowlist=apoc.*

# Memory settings (adjust based on available RAM)
server.memory.heap.initial_size=2g
server.memory.heap.max_size=4g
server.memory.pagecache.size=2g
```

3. **Save the changes** and close the settings window

### Enable APOC Configuration (If Needed)

If APOC procedures are restricted, add this to your settings:

```properties
dbms.security.procedures.allowlist=apoc.*
```

---

## Step 5: Start the Database

1. **Start the Database**:
   - In Neo4j Desktop, click the **"Start"** button next to your database
   - Wait for the status to change to **"Active"** (green indicator)

2. **Note the Connection Details**:
   - **Bolt URL**: `neo4j://localhost:7687` or `bolt://localhost:7687`
   - **HTTP URL**: `http://localhost:7474`
   - **Username**: `neo4j`
   - **Password**: The password you set during database creation

3. **Open Neo4j Browser** (Optional):
   - Click **"Open"** button in Neo4j Desktop
   - Or navigate to `http://localhost:7474` in your web browser
   - Login with username `neo4j` and your password

---

## Step 6: Set Up Python Environment

### Install Python Dependencies

1. **Open Terminal/Command Prompt**:
   - **Windows**: Open PowerShell or Command Prompt
   - **macOS/Linux**: Open Terminal

2. **Navigate to Project Directory**:
   ```bash
   cd path/to/athena-cti-db
   ```

3. **Create a Virtual Environment** (Recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Required Packages**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Required Python Packages

The `requirements.txt` includes:

```txt
neo4j>=5.0.0
psycopg2-binary>=2.9.0
requests>=2.28.0
urllib3>=1.26.0
typing-extensions>=4.0.0
aiohttp>=3.8.0
stix2>=3.0.0
```

> **Note**: `neo4j>=5.0.0` is required when using the Bolt protocol (`neo4j://` or `bolt://`), which is the default. Install it manually if not already present: `pip install neo4j`

---

## Step 7: Configure Database Credentials

### Neo4j Environment Variables

The scripts use environment variables to connect to Neo4j.

**Windows (PowerShell)**:
```powershell
$env:NEO4J_URL="neo4j://127.0.0.1:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="Graph@123"
$env:NEO4J_DB="neo4j"
```

**macOS/Linux (Bash)**:
```bash
export NEO4J_URL="neo4j://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="Graph@123"
export NEO4J_DB="neo4j"
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NEO4J_URL` | `neo4j://127.0.0.1:7687` | Bolt connection URL |
| `NEO4J_USER` | `neo4j` | Database username |
| `NEO4J_PASSWORD` | *(required)* | Database password |
| `NEO4J_DB` | `neo4j` or `athena-cti-db` | Target database name (use `neo4j` for the default DB, or a custom name if you created one) |

---

## Step 8: Run the Population Script

### Navigate to the Script Directory

```bash
cd threat_framework
```

### Run the Script

```bash
python populate_neo4j_complete.py
```

### What Happens During Execution

The script will:

1. **Download Threat Intelligence Data** (skipped automatically if already cached):
   - MITRE ATT&CK framework (git clone)
   - MITRE ENGAGE (git clone)
   - CAPEC (Common Attack Pattern Enumeration and Classification)
   - CWE (Common Weakness Enumeration)
   - CVE (git sparse-checkout, 2024 onwards — large repo)
   - KEV (CISA Known Exploited Vulnerabilities)
   - EPSS scores (downloaded for yesterday's date)

2. **Process and Transform Data**:
   - Parse STIX2 JSON (ATT&CK, ENGAGE), XML (CAPEC, CWE), and JSON (CVE, KEV)
   - Build nodes and relationship queries

3. **Populate Neo4j Database**:
   - Create constraints
   - Insert nodes: Tactics, Techniques, CAPEC, CWE, CVE, KEV, Engage, EPSS
   - Create all intra- and cross-framework relationships

### Expected Output

You should see logs similar to:

```
INFO - Neo4j populate module using Bolt protocol: neo4j://127.0.0.1:7687
INFO - Using Neo4j user: neo4j, database: neo4j
INFO - Starting complete threat intelligence data import...
INFO - Cloning MITRE CTI repository...
INFO - Downloading CAPEC data...
INFO - Downloading and extracting CWE data...
INFO - Cloning CVE repository with sparse checkout (2024 onwards)...
INFO - Downloading KEV data...
INFO - Cloning MITRE Engage repository...
INFO - Creating Neo4j constraints...
INFO - Processing ATT&CK data...
INFO - Creating ATT&CK internal relationships...
INFO - Creating custom ATT&CK relationships...
INFO - Creating derived achieves relationships...
INFO - Processing CAPEC data...
INFO - Creating CAPEC internal relationships...
INFO - Processing CWE data...
INFO - Creating CAPEC external relationships...
INFO - Processing CVE data...
INFO - Progress: Processed 10,000 / XXXXX CVE files (XXXXX valid CVEs)
INFO - Processing KEV data...
INFO - Creating cross-framework relationships...
INFO - Processing Engage data...
INFO - Creating Engage relationships...
INFO - Downloading and updating EPSS data...
INFO - ============================================================
INFO - Import completed successfully!
INFO - ============================================================
```

### Monitor Progress

- The script logs progress regularly
- Large datasets (especially CVE) take significant time
- Progress is logged every 10,000 CVE files processed
- Do not interrupt the process unless necessary

---

## Step 9: Verify the Installation

### Step A: Open Neo4j Browser

1. Navigate to `http://localhost:7474`
2. Login with your credentials

### Step B: Open Neo4j Desktop (Recommended)

1. Click **"Query"** in the sidebar
2. Connect to the local instance containing your database
3. Once connected, run the Cypher queries below

---

## Data Sources

The script automatically downloads data from the following sources:

| Source | URL | Format |
|--------|-----|--------|
| **MITRE ATT&CK** | https://github.com/mitre/cti.git | Git / STIX2 JSON |
| **MITRE ENGAGE** | https://github.com/mitre/engage.git | Git / JSON |
| **CAPEC** | http://capec.mitre.org/data/xml/capec_latest.xml | XML |
| **CWE** | http://cwe.mitre.org/data/xml/cwec_latest.xml.zip | ZIP / XML |
| **CVE** | https://github.com/CVEProject/cvelistV5.git | Git / JSON |
| **CISA KEV** | https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json | JSON Feed |
| **EPSS** | https://epss.cyentia.com/epss_scores-{date}.csv.gz | Gzipped CSV |

### Local Data Directory Structure

Downloaded files are cached under `threat_data/`:

```
threat_data/
├── cti/                                  # MITRE ATT&CK STIX JSON files
├── engage/                               # MITRE ENGAGE JSON files
├── cve/cves/                             # CVE JSON files (organized by year)
├── cwe/                                  # CWE XML data
├── capec_latest.xml                      # CAPEC XML
├── known_exploited_vulnerabilities.json  # CISA KEV
└── epss_scores_data.csv                  # EPSS scores
```

---

## Troubleshooting

### Issue: "Connection refused" or "Unable to connect to Neo4j"

**Solution**:
- Verify Neo4j Desktop shows database as "Active" (green)
- Check the Bolt URL is correct: `neo4j://127.0.0.1:7687`
- Ensure no firewall is blocking port 7687
- Try restarting the database in Neo4j Desktop

### Issue: "Authentication failed"

**Solution**:
- Verify the password in environment variables matches the database password
- Reset password in Neo4j Desktop if needed (Settings → Reset Password)
- Ensure `NEO4J_USER` is set to `neo4j` (default username)

### Issue: "Memory errors" or "Java heap space"

**Solution**:
- Increase heap memory in database settings (Step 4)
- Close other memory-intensive applications
- Restart Neo4j Desktop and try again
- Consider processing data in smaller batches

### Issue: "APOC procedures not found"

**Solution**:
- Verify APOC is installed (Step 3)
- Ensure APOC is enabled in settings:
  ```properties
  dbms.security.procedures.unrestricted=apoc.*
  ```
- Restart the database after installing APOC

### Issue: "Script fails during CVE download"

**Solution**:
- Check internet connection
- The CVE file is very large (500MB+), ensure sufficient bandwidth
- If download fails, manually download from the URL in the script
- Place the file in `threat_data/cve/` directory

### Issue: "Script runs but no data appears in Neo4j"

**Solution**:
- Check script logs for errors
- Verify constraints were created successfully
- Run validation queries in Neo4j Browser
- Check if `threat_data/` directory contains downloaded files
- Ensure sufficient disk space

### Issue: "Python module not found"

**Solution**:
- Ensure virtual environment is activated
- Reinstall requirements:
  ```bash
  pip install --upgrade -r requirements.txt
  ```
- For Bolt protocol, ensure `neo4j` driver is installed:
  ```bash
  pip install neo4j>=5.0.0
  ```

### Issue: "Script is very slow"

**Solution**:
- This is normal for large datasets (CVE has 200K+ entries)
- Expected total time: 30-90 minutes depending on system
- Increase memory allocation in Neo4j settings
- Use SSD instead of HDD for better performance
- Ensure batch_size is appropriate (default: 100)

---

## Performance Tips

1. **Increase Memory**: Allocate 4-8GB heap memory for large datasets
2. **Use SSD**: Store Neo4j data on SSD for faster I/O
3. **Close Other Apps**: Free up system resources during import
4. **Monitor Logs**: Watch for errors or warnings during processing
5. **Batch Processing**: The script processes data in batches (default: 100)

---

## Next Steps

After successful installation:

1. **Explore the Graph**:
   - Use Neo4j Browser to visualize relationships
   - Try sample queries from the verification section

2. **Query the Database**:
   - Build custom Cypher queries for threat intelligence
   - Analyze attack patterns, techniques, and vulnerabilities

3. **Integrate with Applications**:
   - Use Neo4j drivers to query from your applications
   - Build threat intelligence dashboards
   - Create automated security assessments

4. **Keep Data Updated**:
   - Re-run the population script periodically to refresh threat data
   - Monitor MITRE and CISA for new releases

---

## Additional Resources

- **Neo4j Documentation**: https://neo4j.com/docs/
- **Cypher Query Language**: https://neo4j.com/docs/cypher-manual/
- **APOC Documentation**: https://neo4j.com/labs/apoc/
- **MITRE ATT&CK**: https://attack.mitre.org/
- **MITRE ENGAGE**: https://engage.mitre.org/
- **CISA KEV**: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
- **NVD (National Vulnerability Database)**: https://nvd.nist.gov/
- **EPSS**: https://www.first.org/epss/
- **Neo4j Graph Academy**: https://graphacademy.neo4j.com/ (Free courses)

