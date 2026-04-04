#!/bin/bash

# Setup runpod pod for IFT work

if [ "$#" -lt 0 ]; then
    echo "ERROR: missing arguments"
    exit 1
fi

BINDIR=/root/bin
CONFDIR=$BINDIR

# Start the ssh agent in the background:
#eval "$(ssh-agent -s)"

# Add private key to the agent:
#ssh-add ~/.ssh/id_ed25519

# add github server host keys:
#ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts

WORKDIR=/workspace
cd $WORKDIR

HFTOKEN=$(cat ${CONFDIR}/HF-token) 


sudo apt update
sudo apt install less screen -y


# git setup:
git config --global user.email "icardei@fau.edu"

echo 
echo 
echo "Cloning Sophia repo from github"
git  clone https://github.com/Athena-Software-Group/Sophia-tmpl_gen.git || exit 2

# Now run all .sh scripts from $SETUPDIR 
SETUPDIR="${WORKDIR}/Sophia-tmpl_gen/tmpl_gen/ift-pod/setup"

echo
echo
echo "Executing scripts in ${SETUPDIR} ..."

# Use nullglob so the loop doesn't run if no .sh files are found
shopt -s nullglob

for f in "$SETUPDIR"/*.sh; do
    if [[ -f "$f" && -x "$f" ]]; then
        echo "Executing: $f"    
	if ! "$f"; then
            echo "------------------------------------------------"
            echo "CRITICAL: $f failed with exit code $?."
            echo "Terminating sequence."
            echo "------------------------------------------------"
	    FAILMSG="Task ${f} failed."
            break
	fi
    else
        echo "Skipping: $f (Not a file or not executable)"
    fi
done

echo
echo

if [[ -z "${FAILMSG}" ]]; then
    
    echo "All tasks complete."
else
    echo "${FAILMSG}"
    exit 3
fi
