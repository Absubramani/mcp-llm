BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
 
echo "Starting scheduler..."
python "$BASE_DIR/scheduler.py" &
SCHED_PID=$!
echo "Scheduler PID: $SCHED_PID"
 
echo "Starting app..."
streamlit run "$BASE_DIR/app.py"
 
kill $SCHED_PID 2>/dev/null
echo "Stopped."