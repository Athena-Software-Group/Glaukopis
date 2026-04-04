#!/bin/bash

# Setup runpod pod for IFT work

if [ "$#" -lt 3 ]; then
    echo "ERROR: missing arguments"
    echo "Usage: $0 root@<SSH_IP_ADDRESS> -p <SSH_PORT>"
    exit 1
fi

POD_SETUP="pod-setup.sh"

SSHUSERHOST="$1"
SSHPORT="$3"

BINDIR=/root/bin
CONFDIR=$BINDIR

WORKDIR=/workspace

GITHUB_TOKEN=$(cat ${ASGHOME}/work/cloud-stuff/github-athena-pat2)

echo "Generating .netrc for github access"
ssh -T -p ${SSHPORT} ${SSHUSERHOST} "cat > /root/.netrc" <<EOF
machine github.com
login icardei@fau.edu
password ${GITHUB_TOKEN}
EOF

echo "chmod .netrc..."
ssh -T -p ${SSHPORT} ${SSHUSERHOST} "chmod 0600 /root/.netrc"

echo "Creating ${BINDIR} and ${CONFDIR}..."
ssh -T -p ${SSHPORT} ${SSHUSERHOST} <<EOF
mkdir -p $BINDIR
mkdir -p $CONFDIR
EOF


#echo "scp -P ${SSHPORT} $HOME/.ssh/id_ed25519_runpod ${SSHUSERHOST}:/root/.ssh/id_ed25519"
#echo "scp -P ${SSHPORT} $HOME/.ssh/id_ed25519_runpod.pub ${SSHUSERHOST}:/root/.ssh/id_ed25519.pub"

echo "Copy other files..."
scp -P ${SSHPORT} ${ASGHOME}/work/cloud-stuff/{github-athena-pat2,HF-token} ${SSHUSERHOST}:${CONFDIR}
#echo "scp -P ${SSHPORT} ${ASGHOME}/work/cloud-stuff/HF-token ${SSHUSERHOST}:${CONFDIR}"
scp -P ${SSHPORT} ${POD_SETUP} ${SSHUSERHOST}:${BINDIR}

PODSETUP_SH="${BINDIR}/${POD_SETUP}"

echo
echo "Run ${PODSETUP_SH} script to complete system setup and run IFT"

# Uncomment next to run pod-setup script:
# ssh -T -p ${SSHPORT} ${SSHUSERHOST} "${PODSETUP_SH}"
