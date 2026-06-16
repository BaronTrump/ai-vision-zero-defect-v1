#!/usr/bin/env bash
set -e

MODE="${1:-help}"
PROJECT="${2:-all}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$MODE" in
  docker)
    echo -e "${BLUE}🐳 Building & running all projects with Docker...${NC}"
    cd "$ROOT_DIR"
    docker compose up --build -d
    echo ""
    echo -e "${GREEN}✅ All projects started!${NC}"
    echo "   Project 1 (Defect Detector):  http://localhost:8501"
    echo "   Project 2 (Robot Vision):     http://localhost:8502"
    echo "   Project 3 (Anomaly Detector): http://localhost:8503"
    echo "   Project 4 (Monitor Dashboard): http://localhost:8504"
    echo ""
    echo -e "${YELLOW}View logs:${NC} docker compose logs -f"
    echo -e "${YELLOW}Stop all:${NC}  docker compose down"
    ;;

  docker-stop)
    echo -e "${BLUE}🛑 Stopping all Docker containers...${NC}"
    cd "$ROOT_DIR"
    docker compose down
    echo -e "${GREEN}Done.${NC}"
    ;;

  local-install)
    echo -e "${BLUE}📦 Installing dependencies for all projects...${NC}"
    for dir in "$ROOT_DIR"/*/; do
      if [ -f "$dir/requirements.txt" ]; then
        proj=$(basename "$dir")
        echo -e "${YELLOW}Installing $proj...${NC}"
        pip install -r "$dir/requirements.txt"
      fi
    done
    echo -e "${GREEN}✅ All dependencies installed!${NC}"
    ;;

  run)
    case "$PROJECT" in
      1|defect)
        echo -e "${BLUE}🔍 Starting Defect Detector...${NC}"
        cd "$ROOT_DIR/01-ai-vision-defect-detector"
        python src/data_generator.py --samples 200 2>/dev/null || true
        streamlit run src/web_demo.py
        ;;
      2|robot)
        echo -e "${BLUE}🤖 Starting Robot Vision Guidance...${NC}"
        cd "$ROOT_DIR/02-robot-vision-guidance"
        streamlit run src/web_demo.py
        ;;
      3|anomaly)
        echo -e "${BLUE}🔬 Starting Anomaly Detector...${NC}"
        cd "$ROOT_DIR/03-production-anomaly-detector"
        python src/data_generator.py 2>/dev/null || true
        streamlit run src/dashboard.py
        ;;
      4|monitor)
        echo -e "${BLUE}🏭 Starting Monitor Dashboard...${NC}"
        cd "$ROOT_DIR/04-production-monitor-dashboard"
        streamlit run src/dashboard.py
        ;;
      all)
        echo -e "${BLUE}Launching all projects in background...${NC}"
        cd "$ROOT_DIR/01-ai-vision-defect-detector"
        python src/data_generator.py --samples 100 2>/dev/null || true
        streamlit run src/web_demo.py --server.port=8501 &
        cd "$ROOT_DIR/02-robot-vision-guidance"
        streamlit run src/web_demo.py --server.port=8502 &
        cd "$ROOT_DIR/03-production-anomaly-detector"
        python src/data_generator.py 2>/dev/null || true
        streamlit run src/dashboard.py --server.port=8503 &
        cd "$ROOT_DIR/04-production-monitor-dashboard"
        streamlit run src/dashboard.py --server.port=8504 &
        echo -e "${GREEN}✅ All projects starting...${NC}"
        echo "   Project 1: http://localhost:8501"
        echo "   Project 2: http://localhost:8502"
        echo "   Project 3: http://localhost:8503"
        echo "   Project 4: http://localhost:8504"
        echo -e "${YELLOW}Press Ctrl+C to stop all.${NC}"
        wait
        ;;
    esac
    ;;

  generate-data)
    echo -e "${BLUE}📊 Generating training data...${NC}"
    case "$PROJECT" in
      1|defect)
        cd "$ROOT_DIR/01-ai-vision-defect-detector"
        python src/data_generator.py --samples 500
        ;;
      3|anomaly)
        cd "$ROOT_DIR/03-production-anomaly-detector"
        python src/data_generator.py
        ;;
      all)
        cd "$ROOT_DIR/01-ai-vision-defect-detector"
        python src/data_generator.py --samples 200
        cd "$ROOT_DIR/03-production-anomaly-detector"
        python src/data_generator.py
        ;;
    esac
    echo -e "${GREEN}✅ Data generated!${NC}"
    ;;

  train)
    echo -e "${BLUE}🧠 Training models...${NC}"
    cd "$ROOT_DIR/03-production-anomaly-detector"
    python src/train.py
    echo -e "${GREEN}✅ Training complete!${NC}"
    ;;

  help|*)
    echo -e "${BLUE}[Company Name] — AI Vision Project Manager${NC}"
    echo ""
    echo "Usage: ./run.sh <command> [project]"
    echo ""
    echo "Commands:"
    echo "  docker            Build & run all projects in Docker containers"
    echo "  docker-stop       Stop all Docker containers"
    echo "  local-install     Install all Python dependencies locally"
    echo "  run [project]     Run a project locally (1|2|3|4|all|defect|robot|anomaly|monitor)"
    echo "  generate-data [n]  Generate synthetic training data"
    echo "  train             Train anomaly detection model"
    echo "  help              Show this help"
    echo ""
    echo "Examples:"
    echo "  ./run.sh docker              # Run everything in containers"
    echo "  ./run.sh run 1               # Run defect detector locally"
    echo "  ./run.sh run all             # Run all projects locally"
    echo "  ./run.sh generate-data       # Generate training data"
    echo ""
    echo "Quick URL reference (Docker mode):"
    echo "  Defect Detector:   http://localhost:8501"
    echo "  Robot Vision:      http://localhost:8502"
    echo "  Anomaly Detector:  http://localhost:8503"
    echo "  Monitor Dashboard: http://localhost:8504"
    ;;
esac
