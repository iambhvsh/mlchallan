from flask import Flask, render_template, request, jsonify, send_from_directory
import os, json, uuid, base64
import argparse
from datetime import datetime
import cv2
import numpy as np
from utils.detector import TrafficViolationDetector
from utils.challan import ChallanManager

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

detector = TrafficViolationDetector()
challan_mgr = ChallanManager()
RECENT_VIOLATIONS = {}
VIOLATION_COOLDOWN_SECONDS = 30


def _normalize_plate(plate_text: str) -> str:
    return (plate_text or "").upper().replace(" ", "").replace("-", "")


def _to_bool(v) -> bool:
    return str(v).strip().lower() in {'1', 'true', 'yes', 'on'}


def _should_issue_challan(plate_text: str, violation_type: str) -> bool:
    plate = _normalize_plate(plate_text)
    if not plate:
        return False
    key = (plate, violation_type)
    now = datetime.now()
    last_time = RECENT_VIOLATIONS.get(key)
    if last_time and (now - last_time).total_seconds() < VIOLATION_COOLDOWN_SECONDS:
        return False
    RECENT_VIOLATIONS[key] = now
    return True


def _create_challans_from_result(result: dict, violation_type: str, image_filename: str) -> list:
    challans_created = []
    for violation in result.get('violations', []):
        plate = violation.get('plate_text', '')
        if not _should_issue_challan(plate, violation_type):
            continue
        challan = challan_mgr.create_challan(
            plate_text=plate,
            violation_type=violation_type,
            image_path=image_filename,
            confidence=violation.get('plate_confidence', 0),
            vehicle_type=violation.get('vehicle_type', 'Unknown'),
            annotated_image=result.get('annotated_image', image_filename)
        )
        challans_created.append(challan)
    return challans_created

@app.route('/')
def dashboard():
    challans = challan_mgr.get_all_challans()
    stats = challan_mgr.get_stats()
    return render_template('dashboard.html', challans=challans, stats=stats)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'GET':
        return render_template('analyze.html')
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    violation_type = request.form.get('violation_type', 'signal')
    stop_line_y = int(request.form.get('stop_line_y', 50))  # % from top
    auto_stop_line = _to_bool(request.form.get('auto_stop_line', 'false'))

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result = detector.analyze_image(filepath, violation_type, stop_line_y, auto_stop_line=auto_stop_line)

    challans_created = _create_challans_from_result(result, violation_type, filename)

    return jsonify({
        'success': True,
        'detections': result.get('detections', []),
        'violations': result.get('violations', []),
        'challans_created': challans_created,
        'annotated_image': result.get('annotated_image', filename),
        'summary': result.get('summary', ''),
        'stop_line_y': result.get('stop_line_y')
    })

@app.route('/api/live_analyze', methods=['POST'])
def live_analyze():
    data = request.get_json(silent=True) or {}
    image_data = data.get('image', '')
    violation_type = data.get('violation_type', 'signal')
    stop_line_y = int(data.get('stop_line_y', 50))
    stream_id = data.get('stream_id', 'default')
    auto_stop_line = _to_bool(data.get('auto_stop_line', False))

    if not image_data or ',' not in image_data:
        return jsonify({'error': 'Invalid frame payload'}), 400

    try:
        _, encoded = image_data.split(',', 1)
        image_bytes = base64.b64decode(encoded)
    except Exception:
        return jsonify({'error': 'Could not decode image data'}), 400

    npbuf = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(npbuf, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Could not decode frame image'}), 400

    result = detector.analyze_live_frame(
        frame, violation_type, stop_line_y, stream_id=stream_id, auto_stop_line=auto_stop_line
    )

    evidence_filename = ''
    if result.get('violations'):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        evidence_filename = f"live_{uuid.uuid4().hex}.jpg"
        evidence_path = os.path.join(app.config['UPLOAD_FOLDER'], evidence_filename)
        cv2.imwrite(evidence_path, frame)
        result['annotated_image'] = evidence_filename

    challans_created = _create_challans_from_result(result, violation_type, evidence_filename)

    return jsonify({
        'success': True,
        'detections': result.get('detections', []),
        'violations': result.get('violations', []),
        'challans_created': challans_created,
        'annotated_image': result.get('annotated_image', ''),
        'summary': result.get('summary', ''),
        'stop_line_y': result.get('stop_line_y'),
        'live': True
    })

@app.route('/api/live_reset', methods=['POST'])
def live_reset():
    data = request.get_json(silent=True) or {}
    stream_id = data.get('stream_id')
    detector.reset_live_state(stream_id)
    return jsonify({'success': True})

@app.route('/challan/<challan_id>')
def view_challan(challan_id):
    challan = challan_mgr.get_challan(challan_id)
    if not challan:
        return "Challan not found", 404
    return render_template('challan_detail.html', challan=challan)

@app.route('/api/challans')
def api_challans():
    return jsonify(challan_mgr.get_all_challans())

@app.route('/api/stats')
def api_stats():
    return jsonify(challan_mgr.get_stats())

@app.route('/api/mark_paid/<challan_id>', methods=['POST'])
def mark_paid(challan_id):
    data = challan_mgr._read_db()
    for c in data:
        if c['id'] == challan_id:
            c['status'] = 'Paid'
            break
    challan_mgr._write_db(data)
    return jsonify({'success': True})

@app.route('/api/lookup_plate', methods=['POST'])
def lookup_plate():
    plate = request.json.get('plate', '')
    result = challan_mgr.lookup_plate_in_db(plate)
    return jsonify(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ATCS Flask app')
    parser.add_argument('--host', default='0.0.0.0', help='Host interface to bind')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    args = parser.parse_args()

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host=args.host, port=args.port, debug=args.debug)
