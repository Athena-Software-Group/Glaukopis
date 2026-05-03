# Short shell script to install the CTI DB

# 1. Python environment
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
./install.sh

# 2. Neo4j connection parameters
export NEO4J_URL="neo4j://127.0.0.1:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="neo4jneo4j"
export NEO4J_DB="athena-threat-db2"

# 3. Populate the graph
./populate.sh

