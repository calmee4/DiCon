#!/bin/bash

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# ASCII Art for DiCon
echo -e "${BLUE}${BOLD}"
cat << "EOF"
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║     ██████╗ ██╗ ██████╗ ██████╗ ███╗   ██╗                            ║
    ║     ██╔══██╗██║██╔════╝██╔═══██╗████╗  ██║                            ║
    ║     ██║  ██║██║██║     ██║   ██║██╔██╗ ██║                            ║
    ║     ██║  ██║██║██║     ██║   ██║██║╚██╗██║                            ║
    ║     ██████╔╝██║╚██████╗╚██████╔╝██║ ╚████║                            ║
    ║     ╚═════╝ ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝                            ║
    ║                                                                        ║
    ║      🔬 Dual Drug Interaction and Contextual Pre-training 💊          ║
    ║            🏥 for Effective Medication Recommendation 🏥              ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Test MIMIC-III
echo -e "${GREEN}${BOLD}Testing MIMIC-III Best Model${NC}"
python main_OurModel.py --cuda 0 --dataset "mimic-iii" --dim 1536 --Test true --resume_path "./saved/mimic-iii/OurModel/Epoch_6_JA_0.5537_DDI_0.07087.model"

echo ""

# Test MIMIC-IV
echo -e "${GREEN}${BOLD}Testing MIMIC-IV Best Model${NC}"
python main_OurModel.py --cuda 0 --dataset "mimic-iv" --dim 1536 --Test true --resume_path "./saved/mimic-iv/OurModel/Epoch_7_JA_0.4692_DDI_0.07334.model"

echo -e "${YELLOW}${BOLD}DiCon Testing Complete!${NC}"