import os
import csv
import random
from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

# HTML template with explicit "Cull" and "Keep" buttons.
template = """
<!DOCTYPE html>
<html>
  <head>
    <title>IML Cull Labeler</title>
    <style>
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
        max-width: 600px;
      }
      .stat-item {
        text-align: center;
        padding: 0 15px;
      }
      .stat-count {
        font-size: 1.5em;
        font-weight: bold;
      }
      .buttons {
        margin-top: 15px;
      }
      button {
        margin-right: 5px;
        padding: 8px 12px;
      }
    </style>
  </head>
  <body>
    <div class="instruction">
      <p><strong>Note:</strong> You don't need to label all images. You can navigate through images and only label the ones you want.</p>
    </div>
    
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
    
    <h1>Image {{ index+1 }} of {{ total }}</h1>
    <img src="{{ image_url }}" alt="Image" style="max-height:384px;"><br><br>
    <p>Current label: <strong>{{ label if label else "None" }}</strong></p>
    
    <div class="buttons">
      <form method="post">
        <button name="action" value="prev">Prev</button>
        <button name="action" value="next">Next</button>
        <button name="action" value="random">Random</button>
        <button name="action" value="cull">Cull</button>
        <button name="action" value="keep">Keep</button>
        <button name="action" value="remove">Remove Label</button>
      </form>
    </div>
  </body>
</html>
"""

# Global state variables.
root_dir = None
src_dir_name = 'src'  # Default source directory name
labels = {}    # {image_name: "cull" or "keep" or None}
images = []    # list of image filenames
current_index = 0

def load_data(root, source='src'):
    global root_dir, src_dir_name, images, labels, current_index
    root_dir = root
    src_dir_name = source
    src_dir = os.path.join(root_dir, src_dir_name)
    images = sorted(os.listdir(src_dir))
    csv_path = os.path.join(root_dir, 'cull_labels.csv')
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            labels = {row['image_name']: row['label'] for row in reader}
    # No default labels are assigned
    current_index = 0

def save_labels():
    csv_path = os.path.join(root_dir, 'cull_labels.csv')
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

def run_label_app(root, source='src'):
    load_data(root, source)
    app.static_folder = os.path.join(root, source)
    app.run(debug=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run the image labeling interface')
    parser.add_argument('--input', required=True, help='Root directory containing the source folder with images')
    parser.add_argument('--source', default='src', help='Name of the source directory containing images (default: src)')
    args = parser.parse_args()
    run_label_app(args.input, args.source)
