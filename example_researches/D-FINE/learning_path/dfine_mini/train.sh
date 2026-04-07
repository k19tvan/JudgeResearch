#!/bin/bash

# Training script for DFINE object detection model
# Usage: ./train.sh [option]
# Run with no arguments to see all available options

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_help() {
    cat << EOF
${BLUE}=== DFINE Training Script ===${NC}

Usage: ./train.sh [OPTION]

${YELLOW}Predefined Training Scenarios:${NC}
  quick         Run 1 epoch for testing
  test          Run 3 epochs with batch_size=8 (quick validation)
  standard      Run 5 epochs with default settings
  long          Run 10 epochs with optimized settings
  production    Run 20 epochs with larger batch size and optimized lr
  small         Train small model (resource-constrained)
  large         Train large model (better performance)
  cpu           Training on CPU

${YELLOW}Custom Options:${NC}
  custom        Run with custom arguments (pass after this option)
  --help        Show this help message

${YELLOW}Examples:${NC}
  ./train.sh quick
  ./train.sh production
  ./train.sh cpu
  ./train.sh custom --num_epochs 15 --batch_size 16 --learning_rate 5e-5
  python train.py --num_epochs 1 --batch_size 8

${YELLOW}Configuration Files:${NC}
  - TRAINING_GUIDE.md: Detailed documentation of all arguments
  - train.py: Main training script

EOF
}

check_environment() {
    # Check if conda environment is available
    if ! command -v conda &> /dev/null; then
        echo -e "${RED}Error: conda not found${NC}"
        echo "Please install conda or activate the conda environment manually"
        exit 1
    fi
    
    # Check if we're in the conda environment
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        echo -e "${YELLOW}Warning: No conda environment activated. Attempting to activate 'env'...${NC}"
        source activate env || conda activate env
    fi
    
    echo -e "${GREEN}✓ Environment ready${NC}"
}

run_quick() {
    echo -e "${BLUE}Starting QUICK training (1 epoch)...${NC}"
    python train.py \
        --num_epochs 1 \
        --batch_size 8 \
        --num_workers 4 \
        --save_dir ./checkpoints/quick_test \
        --verbose
}

run_test() {
    echo -e "${BLUE}Starting TEST training (3 epochs)...${NC}"
    python train.py \
        --num_epochs 3 \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --num_workers 4 \
        --save_dir ./checkpoints/test_run \
        --verbose
}

run_standard() {
    echo -e "${BLUE}Starting STANDARD training (5 epochs)...${NC}"
    python train.py \
        --num_epochs 5 \
        --batch_size 8 \
        --learning_rate 1e-4 \
        --weight_decay 1e-4 \
        --num_workers 4 \
        --save_dir ./checkpoints/standard \
        --save_interval 1
}

run_long() {
    echo -e "${BLUE}Starting LONG training (10 epochs)...${NC}"
    python train.py \
        --num_epochs 10 \
        --batch_size 16 \
        --learning_rate 5e-5 \
        --weight_decay 1e-4 \
        --num_workers 8 \
        --save_dir ./checkpoints/long_training \
        --save_interval 2 \
        --verbose
}

run_production() {
    echo -e "${BLUE}Starting PRODUCTION training (20 epochs)...${NC}"
    python train.py \
        --num_epochs 20 \
        --batch_size 16 \
        --learning_rate 1e-4 \
        --weight_decay 1e-4 \
        --num_workers 8 \
        --d_model 256 \
        --num_encoder_layers 3 \
        --num_decoder_layers 4 \
        --save_dir ./checkpoints/production \
        --save_interval 2 \
        --verbose
}

run_small() {
    echo -e "${BLUE}Starting SMALL MODEL training (resource-constrained)...${NC}"
    python train.py \
        --num_epochs 5 \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --num_workers 2 \
        --d_model 64 \
        --num_encoder_layers 1 \
        --num_decoder_layers 2 \
        --num_queries 25 \
        --save_dir ./checkpoints/small_model \
        --verbose
}

run_large() {
    echo -e "${BLUE}Starting LARGE MODEL training (better performance)...${NC}"
    python train.py \
        --num_epochs 10 \
        --batch_size 32 \
        --learning_rate 5e-5 \
        --num_workers 8 \
        --d_model 256 \
        --num_encoder_layers 6 \
        --num_decoder_layers 6 \
        --num_queries 100 \
        --save_dir ./checkpoints/large_model \
        --save_interval 2 \
        --verbose
}

run_cpu() {
    echo -e "${BLUE}Starting CPU training...${NC}"
    python train.py \
        --num_epochs 2 \
        --batch_size 4 \
        --learning_rate 1e-4 \
        --num_workers 2 \
        --device cpu \
        --save_dir ./checkpoints/cpu_training \
        --verbose
}

run_custom() {
    echo -e "${BLUE}Running custom training with arguments:${NC}"
    echo -e "${YELLOW}$@${NC}\n"
    python train.py "$@"
}

# Main script logic
main() {
    if [[ $# -eq 0 ]]; then
        print_help
        exit 0
    fi
    
    case "$1" in
        quick)
            check_environment
            run_quick
            ;;
        test)
            check_environment
            run_test
            ;;
        standard)
            check_environment
            run_standard
            ;;
        long)
            check_environment
            run_long
            ;;
        production)
            check_environment
            run_production
            ;;
        small)
            check_environment
            run_small
            ;;
        large)
            check_environment
            run_large
            ;;
        cpu)
            check_environment
            run_cpu
            ;;
        custom)
            check_environment
            shift
            run_custom "$@"
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo ""
            print_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
