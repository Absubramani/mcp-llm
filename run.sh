BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$BASE_DIR/.env" ]; then
    echo "ERROR: .env file not found!"
    exit 1
fi

if [ -f "$BASE_DIR/venv/bin/activate" ]; then
    source "$BASE_DIR/venv/bin/activate"
    echo "Virtual environment activated."
fi

echo "Starting scheduler..."
python "$BASE_DIR/scheduler.py" &
SCHED_PID=$!

sleep 1
if ! kill -0 $SCHED_PID 2>/dev/null; then
    echo "ERROR: Scheduler failed to start!"
    exit 1
fi
echo "Scheduler PID: $SCHED_PID"

echo "Starting app..."
streamlit run "$BASE_DIR/app.py"

kill $SCHED_PID 2>/dev/null
echo "Stopped."