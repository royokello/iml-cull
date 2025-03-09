import os
import csv
from flask import Flask, render_template_string, request, redirect, url_for

app = Flask(__name__)

# HTML template with explicit "Cull" and "Keep" buttons.
template = """
<!DOCTYPE html>
<html>
  <head>
    <title>IML Cull Labeler</title>
  </head>
  <body>
    <h1>Image {{ index+1 }} of {{ total }}</h1>
    <img src="{{ image_url }}" alt="Image" style="max-width:600px;"><br><br>
    <p>Current label: <strong>{{ label }}</strong></p>
    <form method="post">
      <button name="action" value="prev">Prev</button>
      <button name="action" value="next">Next</button>
      <button name="action" value="cull">Cull</button>
      <button name="action" value="keep">Keep</button>
    </form>
  </body>
</html>
"""

# Global state variables.
root_dir = None
labels = {}    # {image_name: "cull" or "keep"}
images = []    # list of image filenames
current_index = 0

def load_data(root):
    global root_dir, images, labels, current_index
    root_dir = root
    src_dir = os.path.join(root_dir, 'src')
    images = sorted(os.listdir(src_dir))
    csv_path = os.path.join(root_dir, 'cull_labels.csv')
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            labels = {row['image_name']: row['label'] for row in reader}
    else:
        # Default all images to "keep".
        labels = {img: "keep" for img in images}
    current_index = 0

def save_labels():
    csv_path = os.path.join(root_dir, 'cull_labels.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for img in images:
            writer.writerow({'image_name': img, 'label': labels.get(img, "keep")})

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_index
    total = len(images)
    if total == 0:
        return "No images found in the src directory."

    if request.method == 'POST':
        action = request.form.get('action')
        current_image = images[current_index]
        if action == 'cull':
            labels[current_image] = "cull"
        elif action == 'keep':
            labels[current_image] = "keep"
        elif action == 'prev':
            current_index = (current_index - 1) % total
        elif action == 'next':
            current_index = (current_index + 1) % total
        save_labels()
        return redirect(url_for('index'))

    image_name = images[current_index]
    image_url = f"/static/{image_name}"  # Serve images from the provided src folder.
    current_label = labels.get(image_name, "keep")
    return render_template_string(template, image_url=image_url, index=current_index, total=total, label=current_label)

def run_label_app(root):
    load_data(root)
    # Serve images from the user-specified root/src folder.
    app.static_folder = os.path.join(root, 'src')
    app.run(debug=True)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python label.py <root_directory>")
        exit(1)
    run_label_app(sys.argv[1])
