#!/usr/bin/env bash
# =========================================================
# RAG System — one-command launcher
# Usage:   ./start.sh         (start everything)
#          ./start.sh stop    (stop all containers)
#          ./start.sh logs    (tail logs)
#          ./start.sh status  (check service health)
# =========================================================

set -e

cd "$(dirname "$0")"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

CMD="${1:-start}"

check_prereqs() {
  if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}✖ Docker is not installed.${NC} Install from https://docs.docker.com/get-docker/"
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}✖ 'docker compose' plugin missing.${NC} Install Docker Compose v2."
    exit 1
  fi
  if ! docker info 2>&1 | grep -qi 'nvidia'; then
    echo -e "${YELLOW}⚠ NVIDIA runtime not detected in Docker.${NC}"
    echo "   vLLM requires an NVIDIA GPU + nvidia-container-toolkit."
    echo "   Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    echo ""
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
  fi
  mkdir -p qdrant_data uploads
}

case "$CMD" in
  start|up)
    check_prereqs
    echo -e "${GREEN}▶ Building and starting services...${NC}"
    echo "   First run downloads ~30GB (vLLM image + Qwen model). Be patient."
    docker compose up -d --build
    echo ""
    echo -e "${GREEN}✔ Services starting.${NC} vLLM takes 3-5 min to load the model."
    echo ""
    echo "   Watch progress:   ./start.sh logs"
    echo "   Check status:     ./start.sh status"
    echo "   Open UI:          http://localhost:8001"
    echo "   Stop everything:  ./start.sh stop"
    ;;

  stop|down)
    echo -e "${YELLOW}▶ Stopping all services...${NC}"
    docker compose down
    echo -e "${GREEN}✔ Stopped.${NC}"
    ;;

  restart)
    docker compose restart
    ;;

  logs)
    docker compose logs -f --tail=100
    ;;

  status|health)
    echo "─── Container status ────────────────────────────"
    docker compose ps
    echo ""
    echo "─── Service health ──────────────────────────────"
    for svc in qdrant:6333/readyz vllm:8000/health api:8001/services; do
      name="${svc%%:*}"; endpoint="${svc#*:}"
      if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${endpoint}" 2>/dev/null | grep -q "200\|204"; then
        echo -e "  ${GREEN}✔${NC} ${name} (localhost:${endpoint%%/*})"
      else
        echo -e "  ${RED}✖${NC} ${name} (localhost:${endpoint%%/*}) — not ready"
      fi
    done
    ;;

  clean)
    read -rp "This deletes all containers + volumes (vector DB, cache). Continue? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
      docker compose down -v
      rm -rf uploads
      echo -e "${GREEN}✔ Cleaned.${NC} qdrant_data/ is preserved — delete manually if needed."
    fi
    ;;

  *)
    echo "Usage: $0 {start|stop|restart|logs|status|clean}"
    exit 1
    ;;
esac
