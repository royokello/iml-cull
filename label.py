import argparse
import os
import csv
import random
from flask import Flask, render_template_string, request, redirect, url_for, flash
from utils import find_latest_stage

app = Flask(__name__)

root_dir = None
src_dir_name = None
stage_number = None
images = []  # List of image filenames
current_index = 0
labels = {}  # {image_name: {"label_value": 0 or 1, "tag": "specific_label_reason"}}
KEEP_LABELS_OPTS = [] # Populated by command-line args
CULL_LABELS_OPTS = [] # Populated by command-line args

# --- HTML Template ---
template = """
<!DOCTYPE html>
<html>
<head>
    <title>IML Cull</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .header { text-align: center; margin-bottom: 20px; }
        .title { font-size: 2.5em; margin-bottom: 0; }
        .subtitle { color: #777; font-size: 1.2em; margin-top: 5px; }
        .main-content { display: flex; flex-wrap: wrap; gap: 20px; }
        .image-section { flex: 1; min-width: 400px; text-align: center; }
        .controls-section { flex: 1; min-width: 300px; }
        .stats { background-color: #e6f7ff; padding: 10px; margin: 15px 0; border-radius: 5px; display: flex; justify-content: space-around; flex-wrap: wrap; }
        .stat-item { text-align: center; padding: 5px 10px; }
        .stat-count { font-size: 1.5em; font-weight: bold; }
        .image-container img { max-height:500px; border: 3px solid black; }
        .buttons { margin-top: 15px; }
        button { margin-right: 5px; margin-bottom: 5px; padding: 10px 15px; cursor: pointer; border-radius: 4px; border: 1px solid #ccc; }
        .nav-buttons button { background-color: #f0f0f0; }
        .action-buttons button { font-weight: bold; }
        .action-buttons button[value="keep"] { background-color: #d4edda; color: #155724; border-color: #c3e6cb;} /* This class might be unused if we only have specific keep/cull buttons */
        .action-buttons button.keep-label-button { background-color: #d4edda; color: #155724; border-color: #c3e6cb;}
        .action-buttons button.cull-label-button { background-color: #f8d7da; color: #721c24; border-color: #f5c6cb;}
        .action-buttons button[value="remove"] { background-color: #fff3cd; color: #856404; border-color: #ffeeba;}
        .flash-messages { padding: 10px; margin-bottom: 15px; border-radius: 5px; text-align: center; }
        .flash-error { background-color: #f8d7da; color: #721c24; }
        .flash-success { background-color: #d4edda; color: #155724; }
        .flash-warning { background-color: #fff3cd; color: #856404; }
        .manual-nav { margin: 10px 0; display: flex; align-items: center; }
        .manual-nav input { padding: 8px; width: 120px; margin-right: 5px; border-radius: 4px; border: 1px solid #ccc; }
    </style>
    <script>
      document.addEventListener('keydown', function(event) {
        if (event.key.toLowerCase() === 'r') {
          const randomButton = document.querySelector('button[value="random"]');
          if (randomButton) randomButton.click();
        }
        // Add more shortcuts here if needed, e.g., for tags
      });
    </script>
</head>
<body>
    <div class="header">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
            <div class="flash-messages">
                {% for category, message in messages %}
                <div class="flash-{{ category }}">{{ message }}</div>
                {% endfor %}
            </div>
            {% endif %}
        {% endwith %}
        <h1 class="title">IML Cull</h1>
        <p class="subtitle">royokello</p>
    </div>

    <div class="main-content">
        <div class="image-section">
            <div class="image-container">
                <h2>Image {{ index + 1 }} of {{ total }}</h2>
                <img src="{{ image_url }}" alt="Image {{ images[index] if images and index < images|length else 'N/A' }}">
                <p><small>{{ images[index] if images and index < images|length else 'N/A' }}</small></p>
            </div>
        </div>

        <div class="controls-section">
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-count">{{ stats.get("keep_total", 0) }}</div>
                    <div>Keep (Total)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-count">{{ stats.get("cull_total", 0) }}</div>
                    <div>Cull (Total)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-count">{{ stats.get("labeled_total", 0) }}</div>
                    <div>Labeled (Any)</div>
                </div>
                <div class="stat-item">
                    <div class="stat-count">{{ stats.get("unlabeled_total", 0) }}</div>
                    <div>Unlabeled</div>
                </div>
                <div class="stat-item">
                    <div class="stat-count">{{ stats.get("total_images", 0) }}</div>
                    <div>Total Images</div>
                </div>
            </div>

            <p>Current label: <strong>{{ current_display_label if current_display_label else "None" }}</strong></p>

            <form method="post">
                <div class="buttons nav-buttons">
                    <button name="action" value="prev">← Previous</button>
                    <button name="action" value="next">Next →</button>
                    <button name="action" value="random">Random</button>
                </div>
                <div class="manual-nav">
                    <input type="number" name="goto_index" min="1" max="{{ total if total > 0 else 1 }}" placeholder="Enter image #">
                    <button name="action" value="goto">Go</button>
                </div>
                <div class="manual-nav">
                    <input type="text" name="search_query" placeholder="Enter image name/part">
                    <button name="action" value="search_by_name">Search</button>
                </div>
                <div class="buttons action-buttons">
                    <!-- Keep Label Buttons -->
                    {% for label_opt in keep_labels_opts %}
                    <button name="action" value="keep_{{ label_opt }}" class="keep-label-button">Keep: {{ label_opt.replace('_', ' ').title() }} ({{ stats.get("keep_" + label_opt, 0) }})</button>
                    {% endfor %}

                    <!-- Cull Label Buttons -->
                    {% for label_opt in cull_labels_opts %}
                    <button name="action" value="cull_{{ label_opt }}" class="cull-label-button">Cull: {{ label_opt.replace('_', ' ').title() }} ({{ stats.get("cull_" + label_opt, 0) }})</button>
                    {% endfor %}
                    <button name="action" value="remove">Remove Label</button>
                </div>
            </form>
        </div>
    </div>
</body>
</html>
"""

# --- Helper Functions ---
def load_data(project_path, current_stage=None):
    global root_dir, src_dir_name, stage_number, images, labels, current_index
    root_dir = project_path



    if current_stage is None:
        try:
            # Assuming find_latest_stage returns an integer stage number directly for label.py context
            # If it returns a dict {'stage': X, ...}, this needs adjustment: stage_number = find_latest_stage(project_path)['stage']
            stage_number = find_latest_stage(project_path)
        except ValueError as e:
            flash(f"Error finding latest stage: {e}. Please specify a stage.", "error")
            return False
    else:
        stage_number = current_stage

    src_dir_name = f"stage_{stage_number}"
    image_dir_path = os.path.join(root_dir, src_dir_name)

    if not os.path.isdir(image_dir_path):
        flash(f"Error: Stage directory '{image_dir_path}' not found.", "error")
        images = []
        return False

    images = sorted([f for f in os.listdir(image_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    if not images:
        flash(f"No images found in '{image_dir_path}'.", "warning")
        return True # No images, but not a fatal error for app start, just show message

    current_index = 0
    labels = {}
    
    # Load labels from CSV
    csv_path = os.path.join(root_dir, f"stage_{stage_number}_cull_labels.csv")
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                if header == ['image_name', 'label']: # Old format: image_name, label_str ("keep" or "cull")
                    for row in reader:
                        if len(row) == 2:
                            img_name, label_str = row
                            if label_str.lower() in ["keep", "cull"]:
                                labels[img_name] = {"label_value": 0, "tag": None}
                elif header == ['image_path', 'label', 'tag']: # New format, expecting full image paths
                    for row in reader:
                        if len(row) == 3:
                            image_path_from_csv, label_val_csv_str, tag_from_csv = row
                            img_name = os.path.basename(image_path_from_csv) # Use basename as internal key
                            try:
                                label_val_csv = int(label_val_csv_str)
                                normalized_tag_from_csv = tag_from_csv.strip() if tag_from_csv else None

                                if label_val_csv == 0: # CSV 'Keep'
                                    if normalized_tag_from_csv and normalized_tag_from_csv in KEEP_LABELS_OPTS:
                                        labels[img_name] = {"label_value": 0, "tag": normalized_tag_from_csv}
                                    elif normalized_tag_from_csv: # Tag provided but not a valid keep_label
                                        print(f"Warning: Row for '{img_name}' has Keep label (0) but unrecognized keep tag: '{normalized_tag_from_csv}'. Skipping this entry.")
                                        continue
                                    else: # No tag provided for a keep label
                                        print(f"Warning: Row for '{img_name}' has Keep label (0) but missing keep tag. Assuming generic keep if no specific keep labels are defined or if this behavior is desired, otherwise skipping. For now, skipping.")
                                        continue
                                elif label_val_csv == 1: # CSV 'Cull'
                                    if normalized_tag_from_csv and normalized_tag_from_csv in CULL_LABELS_OPTS:
                                        labels[img_name] = {"label_value": 1, "tag": normalized_tag_from_csv}
                                    elif normalized_tag_from_csv: # Tag provided but not a valid cull_label
                                        print(f"Warning: Row for '{img_name}' has Cull label (1) but unrecognized cull tag: '{normalized_tag_from_csv}'. Skipping this entry.")
                                        continue
                                    else: # No tag provided for a cull label
                                        print(f"Warning: Row for '{img_name}' has Cull label (1) but missing cull tag. Skipping this entry.")
                                        continue
                                else:
                                    print(f"Warning: Row for '{img_name}' has unrecognized label value '{label_val_csv_str}' in CSV. Skipping this entry.")
                                    continue

                            except ValueError:
                                print(f"Skipping row for '{img_name}' due to non-integer label value: '{label_val_csv_str}'.")
    return True

def save_labels():
    if root_dir is None or stage_number is None:
        print("Error: Project directory or stage number not set. Cannot save labels.")
        return

    csv_path = os.path.join(root_dir, f"stage_{stage_number}_cull_labels.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label', 'tag'])  # Header for new format
        for image_basename in sorted(labels.keys()): # Internal keys are basenames
            label_info = labels[image_basename]
            tag_value = label_info.get("tag")
            label_to_write_csv = label_info.get("label_value")
            
            # Ensure label_to_write_csv is not None, default to a value if necessary (e.g., 1 for cull if undefined)
            if label_to_write_csv is None:
                print(f"Warning: label_value is None for {image_basename}. Defaulting to 1 (cull). Check labeling logic.")
                label_to_write_csv = 1 # Or handle as an error
            
            # Prepare the tag string for the CSV (empty if tag is None)
            tag_to_write_csv = tag_value if tag_value is not None else ""

            # Construct full image path for CSV output
            # Ensure root_dir and src_dir_name are available and correctly set
            if root_dir and src_dir_name:
                full_image_path = os.path.join(root_dir, src_dir_name, image_basename)
            else:
                # Fallback or error if path components are not set - this shouldn't happen in normal operation
                print(f"Error: root_dir or src_dir_name not set. Cannot construct full path for {image_basename}")
                full_image_path = image_basename # Or skip? For now, write basename as fallback.

            writer.writerow([full_image_path, label_to_write_csv, tag_to_write_csv])

def get_label_stats():
    global images, labels, KEEP_LABELS_OPTS, CULL_LABELS_OPTS
    total_images_count = len(images)
    
    stats = {
        "keep_total": 0,
        "cull_total": 0,
        "labeled_total": 0,
        "unlabeled_total": 0,
        "total_images": total_images_count
    }

    # Initialize counts for specific keep/cull labels
    for kl_opt in KEEP_LABELS_OPTS:
        stats[f"keep_{kl_opt}"] = 0
    for cl_opt in CULL_LABELS_OPTS:
        stats[f"cull_{cl_opt}"] = 0

    for img_name, label_info in labels.items():
        if label_info is not None:
            stats["labeled_total"] += 1
            label_value = label_info.get("label_value")
            tag = label_info.get("tag")

            if label_value == 0:
                stats["keep_total"] += 1
                if tag and tag in KEEP_LABELS_OPTS:
                    stats[f"keep_{tag}"] += 1
            elif label_value == 1:
                stats["cull_total"] += 1
                if tag and tag in CULL_LABELS_OPTS:
                    stats[f"cull_{tag}"] += 1
            
    stats["unlabeled_total"] = total_images_count - stats["labeled_total"]
    return stats

def randomize_image_index():
    global current_index
    total = len(images)
    if total > 1:
        new_index = current_index
        while new_index == current_index:
            new_index = random.randint(0, total - 1)
        current_index = new_index
    elif total == 1:
        current_index = 0 # Stay on the only image

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    global current_index, labels, TAG_COUNTS
    if not images:
        # Attempt to reload data if images list is empty, might happen on first load if project/stage wasn't ready
        if root_dir:
            load_data(root_dir, stage_number, TAG_OPTIONS)
        if not images: # Still no images
             flash("No images loaded. Please check project and stage configuration.", "error")
             return render_template_string(template, image_url=None, index=0, total=0, current_display_label="None", stats={}, tag_options=TAG_OPTIONS, images=None)

    total_num_images = len(images)

    if request.method == 'POST':
        action = request.form.get('action')
        current_image_name = images[current_index]
        previous_label_info = labels.get(current_image_name, {})

        if action.startswith("keep_"):
            chosen_label = action[len("keep_"):]
            if chosen_label in KEEP_LABELS_OPTS:
                labels[current_image_name] = {"label_value": 0, "tag": chosen_label}
                randomize_image_index()
            else:
                flash(f'Unknown keep label: {chosen_label}', 'error')
        elif action.startswith("cull_"):
            chosen_label = action[len("cull_"):]
            if chosen_label in CULL_LABELS_OPTS:
                labels[current_image_name] = {"label_value": 1, "tag": chosen_label}
                randomize_image_index()
            else:
                flash(f'Unknown cull label: {chosen_label}', 'error')
        elif action == 'keep':
            # If previously tagged, decrement that tag's count
            if previous_label_info.get("tag") and previous_label_info["tag"] in TAG_COUNTS:
                TAG_COUNTS[previous_label_info["tag"]] = max(0, TAG_COUNTS[previous_label_info["tag"]] - 1)
            labels[current_image_name] = {"label_value": 0, "tag": None}
            randomize_image_index()
        elif action == 'remove':
            if current_image_name in labels:
                if previous_label_info.get("tag") and previous_label_info["tag"] in TAG_COUNTS:
                    TAG_COUNTS[previous_label_info["tag"]] = max(0, TAG_COUNTS[previous_label_info["tag"]] - 1)
                del labels[current_image_name]
            randomize_image_index()
        elif action == 'prev':
            current_index = (current_index - 1 + total_num_images) % total_num_images if total_num_images > 0 else 0
        elif action == 'next':
            current_index = (current_index + 1) % total_num_images if total_num_images > 0 else 0
        elif action == 'random':
            randomize_image_index()
        elif action == 'goto':
            try:
                goto_idx = int(request.form.get('goto_index', 1)) - 1
                if 0 <= goto_idx < total_num_images:
                    current_index = goto_idx
            except ValueError:
                pass  # Silently ignore invalid number for 'Go to', or consider an error flash if preferred
        elif action == 'search_by_name':
            search_term = request.form.get('search_query', '').strip().lower()
            if search_term:
                found_idx = -1
                # Exact match first
                for idx, img_name in enumerate(images):
                    if search_term == os.path.splitext(img_name)[0].lower():
                        found_idx = idx
                        break
                # Substring match if no exact match
                if found_idx == -1:
                    for idx, img_name in enumerate(images):
                        if search_term in img_name.lower():
                            found_idx = idx
                            break
                if found_idx != -1:
                    current_index = found_idx
                else:
                    flash(f'Image containing "{search_term}" not found.', 'error')
            else:
                pass # Silently ignore empty search term, or consider an error flash
        
        save_labels()
        return redirect(url_for('index'))

    # GET request or after processing POST
    current_image_name = images[current_index] if images and 0 <= current_index < len(images) else None
    image_url = f"/static/{current_image_name}" if current_image_name else None
    
    current_label_info = labels.get(current_image_name, None)
    display_label_str = "None"
    if current_label_info:
        label_val = current_label_info.get("label_value")
        tag_reason = current_label_info.get("tag", "")
        if label_val == 0:
            display_label_str = f"Keep (Reason: {tag_reason.replace('_', ' ').title() if tag_reason else 'General' })"
        elif label_val == 1:
            display_label_str = f"Cull (Reason: {tag_reason.replace('_', ' ').title() if tag_reason else 'General'})"
        # If no specific tag/reason but label_value is set, it might just show Keep/Cull. Current logic ensures a tag.
            
    current_stats = get_label_stats()
    
    return render_template_string(template, 
                                 image_url=image_url, 
                                 index=current_index, 
                                 total=total_num_images, 
                                 current_display_label=display_label_str,
                                 stats=current_stats,
                                 keep_labels_opts=KEEP_LABELS_OPTS,
                                 cull_labels_opts=CULL_LABELS_OPTS,
                                 images=images, # Pass full images list for display name
                                 current_label_info=current_label_info
                                 )

def start_labeller(project_path, current_stage=None, keep_tags=None, cull_tags=None):
    global KEEP_LABELS_OPTS, CULL_LABELS_OPTS
    KEEP_LABELS_OPTS = keep_tags
    CULL_LABELS_OPTS = cull_tags
    
    if not load_data(project_path, current_stage):
        # load_data will flash an error message if it fails critically
        # We might want to render a simple error page or exit
        print("Failed to load data. Application cannot start.")
        # For a web app, it's better to show an error in the browser if possible
        # but initial load_data errors are tricky before Flask is fully running.
        # If Flask is running, flash() messages will be displayed.
        # If load_data returns False due to critical error (e.g. no stage dir)
        # we might not even have images to show.
        if not images: # If truly no images and critical error
            app.add_url_rule('/', 'error_page', lambda: "Error loading image data. Please check console and project/stage setup.")
            app.run(debug=True, use_reloader=False)
            return
            
    # Set static folder based on successfully loaded src_dir_name
    if root_dir and src_dir_name:
        app.static_folder = os.path.join(root_dir, src_dir_name)
    else: # Fallback if src_dir_name didn't get set (should be caught by load_data checks)
        app.static_folder = project_path 
        print(f"Warning: src_dir_name is not set. Static files may not serve correctly. Defaulting to project root: {project_path}")

    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5050)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--keep-tags', nargs='+', default=["clear_subject"])
    parser.add_argument('--cull-tags', nargs='+', default=["no_subject"])
    args = parser.parse_args()
    
    start_labeller(args.project, args.stage, args.keep_tags, args.cull_tags)
