
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import time
from threading import Thread, Lock
from werkzeug.utils import secure_filename
from demo import EnhancedLogAnomalyDetector

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variables with thread safety
training_lock = Lock()
training_in_progress = False
current_training_thread = None
training_progress = 0
training_status = "Ready"
detector = EnhancedLogAnomalyDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/results', methods=['GET'])
def get_previous_results():
    try:
        results_path = os.path.join(detector.base_path, "enhanced_results.json")
        
        # If results file doesn't exist, return empty list
        if not os.path.exists(results_path):
            return jsonify([])
            
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        formatted_results = []
        for key, metrics in results.items():
            # Skip entries with errors
            if isinstance(metrics, dict) and 'error' in metrics:
                continue
                
            dataset, model = key.split('_', 1)
            formatted_results.append({
                'dataset': dataset,
                'model': model.replace('_', ' ').title(),
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': metrics.get('f1', 0),
                'roc_auc': metrics.get('roc_auc', 0.5),
                'pr_auc': metrics.get('pr_auc', 0),
                'date': time.strftime('%Y-%m-%d', time.localtime(os.path.getmtime(results_path)))
            })
        
        # Sort by F1 score descending
        formatted_results.sort(key=lambda x: x['f1'], reverse=True)
        return jsonify(formatted_results)
    except Exception as e:
        app.logger.error(f"Error getting results: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def update_training_progress(progress, status):
    global training_progress, training_status
    with training_lock:
        training_progress = progress
        training_status = status

@app.route('/api/train', methods=['POST'])
def start_training():
    global training_in_progress, current_training_thread
    
    if training_in_progress:
        return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400
    
    data = request.json
    dataset = data.get('dataset')
    epochs = int(data.get('epochs', 100))
    batch_size = int(data.get('batch_size', 64))
    
    if not dataset:
        return jsonify({'status': 'error', 'message': 'Missing dataset parameter'}), 400
    
    training_in_progress = True
    update_training_progress(0, f"Starting training for {dataset} dataset...")

    def training_task():
        global training_in_progress
        try:
            # Create a callback function to update progress
            def progress_callback(progress, status):
                update_training_progress(progress, status)
                time.sleep(0.1)  # Prevent UI from updating too fast

            success = detector.train_ensemble(
                dataset, 
                epochs, 
                batch_size,
                progress_callback=progress_callback
            )
            
            if not success:
                update_training_progress(0, "Training failed")
                raise Exception("Training failed")
                
            update_training_progress(100, "Training completed successfully")
        except Exception as e:
            app.logger.error(f"Training error: {str(e)}")
            update_training_progress(0, f"Training error: {str(e)}")
        finally:
            training_in_progress = False
    
    current_training_thread = Thread(target=training_task)
    current_training_thread.start()
    
    return jsonify({
        'status': 'success', 
        'message': 'Training started',
        'dataset': dataset
    })

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    with training_lock:
        return jsonify({
            'in_progress': training_in_progress,
            'progress': training_progress,
            'status': training_status
        })

@app.route('/api/detect', methods=['POST'])
def detect_anomalies():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    try:
        # Secure filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Run detection
        results = detector.detect_anomalies(file_path)
        
        if results is None:
            return jsonify({
                'status': 'error',
                'message': 'Detection failed - no trained model available'
            }), 400
            
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Detection error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Start the application
    app.run(debug=True, threaded=True)