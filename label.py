import os
import csv
import random
from flask import Flask, render_template_string, request, redirect, url_for
from utils import find_latest_stage

app = Flask(__name__)

# HTML template with explicit "Cull" and "Keep" buttons.
template = """
<!DOCTYPE html>
<html>
  <head>
    <title>IML Cull Labeler</title>
    <style>
      body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 20px;
      }
      .header {
        text-align: center;
        margin-bottom: 20px;
      }
      .title {
        font-size: 2.5em;
        margin-bottom: 0;
      }
      .subtitle {
        color: #777;
        font-size: 1.2em;
        margin-top: 5px;
      }
      .main-content {
        display: flex;
        flex-wrap: wrap;
        gap: 20px;
      }
      .image-section {
        flex: 1;
        min-width: 400px;
      }
      .controls-section {
        flex: 1;
        min-width: 300px;
      }
      .instruction {
        background-color: #f0f0f0;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
      }
      .stats {
        background-color: #e6f7ff;
        padding: 10px;
        margin: 15px 0;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
      }
      .stat-item {
        text-align: center;
        padding: 0 15px;
      }
      .stat-count {
        font-size: 1.5em;
        font-weight: bold;
      }
      .image-container {
        text-align: center;
      }
      .buttons {
        margin-top: 15px;
      }
      button {
        margin-right: 5px;
        margin-bottom: 5px;
        padding: 10px 15px;
        cursor: pointer;
      }
      .nav-buttons button {
        background-color: #f0f0f0;
      }
      .action-buttons button {
        font-weight: bold;
      }
      .action-buttons button[value="keep"] {
        background-color: #d4edda;
        color: #155724;
      }
      .action-buttons button[value="cull"] {
        background-color: #f8d7da;
        color: #721c24;
      }
      .action-buttons button[value="remove"] {
        background-color: #fff3cd;
        color: #856404;
      }
    </style>
    <script>
      // Add keyboard shortcuts
      document.addEventListener('keydown', function(event) {
        // 'R' key for randomizing image
        if (event.key.toLowerCase() === 'r') {
          // Find the random button and click it
          const randomButton = document.querySelector('button[value="random"]');
          if (randomButton) {
            randomButton.click();
          }
        }
      });
    </script>
  </head>
  <body>
    <div class="header">
      <h1 class="title">IML Cull</h1>
      <p class="subtitle">royokello</p>
    </div>
    
    <div class="main-content">
      <div class="image-section">
        <div class="image-container">
          <h2>Image {{ index+1 }} of {{ total }}</h2>
          <img src="{{ image_url }}" alt="Image" style="max-height:500px; border: 3px solid black;">
        </div>
      </div>
      
      <div class="controls-section">
        
        <div class="stats">
          <div class="stat-item">
            <div class="stat-count">{{ stats.keep }}</div>
            <div>Keep</div>
          </div>
          <div class="stat-item">
            <div class="stat-count">{{ stats.cull }}</div>
            <div>Cull</div>
          </div>
          <div class="stat-item">
            <div class="stat-count">{{ stats.unlabeled }}</div>
            <div>Unlabeled</div>
          </div>
          <div class="stat-item">
            <div class="stat-count">{{ stats.total }}</div>
            <div>Total</div>
          </div>
        </div>
        
        <p>Current label: <strong>{{ label if label else "None" }}</strong></p>
        
        <form method="post">
          <div class="buttons nav-buttons">
            <button name="action" value="prev">← Previous</button>
            <button name="action" value="next">Next →</button>
            <button name="action" value="random">Random</button>
          </div>
          
          <div class="buttons action-buttons">
            <button name="action" value="keep">KEEP</button>
            <button name="action" value="cull">CULL</button>
            <button name="action" value="remove">Remove Label</button>
          </div>
        </form>
      </div>
    </div>
  </body>
</html>
"""

# Global state variables.
root_dir = None
src_dir_name = None
stage_number = None
labels = {}    # {image_name: "cull" or "keep" or None}
images = []    # list of image filenames
current_index = 0

def load_data(project, stage=None):
    global root_dir, src_dir_name, images, labels, current_index, stage_number
    root_dir = project
    
    # If stage is not provided, try to find the latest stage
    if stage is None:
        try:
            stage_number = find_latest_stage(project)
        except ValueError as e:
            raise ValueError(f"No stage specified and {str(e)}")
    else:
        stage_number = stage
    
    src_dir_name = f"stage_{stage_number}"
    src_dir = os.path.join(root_dir, src_dir_name)
    
    if not os.path.exists(src_dir):
        raise ValueError(f"Stage directory {src_dir_name} does not exist in {root_dir}")
    
    images = sorted(os.listdir(src_dir))
    csv_path = os.path.join(root_dir, f"stage_{stage_number}_cull_labels.csv")
    
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            labels = {row['image_name']: row['label'] for row in reader}
    # No default labels are assigned
    current_index = 0

def save_labels():
    csv_path = os.path.join(root_dir, f"stage_{stage_number}_cull_labels.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for img in images:
            if img in labels and labels[img]:  # Only write labeled images
                writer.writerow({'image_name': img, 'label': labels[img]})

def get_label_stats():
    """Calculate statistics for each label type"""
    total = len(images)
    keep_count = sum(1 for img, label in labels.items() if label == "keep")
    cull_count = sum(1 for img, label in labels.items() if label == "cull")
    unlabeled_count = total - keep_count - cull_count
    
    return {
        "keep": keep_count,
        "cull": cull_count,
        "unlabeled": unlabeled_count,
        "total": total
    }

def randomize_image_index():
    """Select a new random image index different from the current one."""
    global current_index
    total = len(images)
    if total > 1:
        # Ensure we get a different index than the current one
        new_index = current_index
        while new_index == current_index:
            new_index = random.randint(0, total - 1)
        current_index = new_index

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_index
    total = len(images)
    if total == 0:
        return f"No images found in the {src_dir_name} directory."

    if request.method == 'POST':
        action = request.form.get('action')
        current_image = images[current_index]
        if action == 'cull':
            labels[current_image] = "cull"
            # Automatically randomize after labeling
            randomize_image_index()
        elif action == 'keep':
            labels[current_image] = "keep"
            # Automatically randomize after labeling
            randomize_image_index()
        elif action == 'remove':
            if current_image in labels:
                del labels[current_image]
            # Automatically randomize after removing label
            randomize_image_index()
        elif action == 'prev':
            current_index = (current_index - 1) % total
        elif action == 'next':
            current_index = (current_index + 1) % total
        elif action == 'random':
            randomize_image_index()
        save_labels()
        return redirect(url_for('index'))

    image_name = images[current_index]
    image_url = f"/static/{image_name}"  # Images are served from the static folder which is set to the source directory
    current_label = labels.get(image_name, None)
    stats = get_label_stats()
    
    return render_template_string(template, 
                                 image_url=image_url, 
                                 index=current_index, 
                                 total=total, 
                                 label=current_label,
                                 stats=stats)

def run_label_app(project, stage=None):
    load_data(project, stage)
    app.static_folder = os.path.join(project, src_dir_name)
    app.run(debug=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the image labeling interface')
    parser.add_argument('--project', required=True, help='Root directory of the project containing stage folders with images')
    parser.add_argument('--stage', type=int, help='Stage number to label (if not provided, will use the latest stage)')
    args = parser.parse_args()
    run_label_app(args.project, args.stage)
