# Django Webapp for Visualizing Sensorium Dataset

The web application allows users to visualize and explore the Sensorium dataset, which contains video recordings of mouse visual cortex neurons responding to various stimuli. All the documentation about the dataset can be found on the [Sensorium Challenge website](https://www.sensorium-competition.net/).

This is a project part of the Master in Data Management and Curation from AREA Science Park and SISSA in the course of AppliedAI.

## Contributors

- Enrico
- Emmanuel
- Ana
- Ruth
- Smriti
- Valentin
- Lesly
- Luis

## Run the Django Webapp

Provided that you have setup the environment and downloaded the dataset as described below, you can locally run the webapp from the root directory of the project  by executing:

```bash
python scripts/manage.py runserver 0.0.0:8050 # or whatever port you prefer
```

Click on the link printed in the terminal to open the webapp in your browser.

If it doesn't open automatically, you can manually navigate to `http://localhost:8050` or `http://0.0.0.0:8050` in your web browser


## Environment Setup

1. Download and install the `uv` package manager with the following commands:

```bash
curl -LsSf https://astral.sh/uv/install.sh -o install_uv.sh # Download installation script
sh install_uv.sh # Run the installation script
rm install_uv.sh # Clean up installation script
```

2. Create, activate and set up a new environment:

```bash
uv venv myenv # Create a new environment named 'myenv'
source myenv/bin/activate # Activate the environment
uv sync --active # Synchronize the environment with the specified dependencies
```

3. To deactivate the environment when done, use:

```bash
deactivate # Deactivate the current environment
```

Using the python notebooks and **provided that you have the relevant extensions installed**, you can also select the created environment `myenv` as the kernel for your notebooks.


## Dataset

The dataset is about the video recordings of mouse visual cortex neurons responding to natural and synthetic stimuli.

### Downloading the Data

Since the dataset is large (50GB) we recommend following the instructions from the [Sensorium Challenge](https://www.sensorium-competition.net/) to download the data. direct link to the github repository containing the data is https://gin.g-node.org/pollytur/sensorium_2023_dataset
All the data need to be saved at the root of this repository in a folder named `project_data`.

## Project Structure

```
Sensorium/
├── webapp/                    # Single directory containing ALL webapp files
│   ├── settings.py            # Django settings (apps, static files, etc.)
│   ├── project_urls.py        # Main URL routing (root URLconf)
│   ├── wsgi.py                # WSGI configuration (for deployment)
│   ├── asgi.py                # ASGI configuration (for async deployment)
│   ├── views.py               # View functions + API endpoints
│   ├── urls.py                # App-specific URL routing
│   ├── apps.py                # App configuration
│   ├── data_loader.py         # MouseDataManager class
│   ├── loader.py              # SensoriumSession class
│   ├── db.sqlite3             # SQLite database (if used)
│   ├── utils/
│   │   └── video_converter.py # Video conversion utilities
│   ├── templates/
│   │   └── webapp/
│   │       └── index.html     # Main HTML template
│   └── static/
│       └── webapp/
│           ├── css/
│           │   └── style.css  # Stylesheet
│           └── js/
│               ├── main.js    # Main JavaScript (AJAX, event handlers)
│               └── plotting.js # Plotly.js plotting functions
│
├── scripts/                   # Scripts and notebooks
│   ├── manage.py              # Django management script
│   └── create_combined_metadata.py # Metadata creation script
│
├── project_data/              # Mouse data directories
├── results/                   # Processed metadata JSON files
```
## Naming and Duplicates Scripts
A script for labelling the video has been implemented. One video label remained unknown, that was labelled manually after visual inspection. 
One more way to perform duplicates detection was pBy applying Transfert learning(R(2+1))-18 model) we computed embedding(each video =512-vector), where we then computed the cosine similarities, and set a hard threshold=1, we flagged out near(exact) duplicates.). 

## License
This project is licensed under the MIT License - see the LICENSE file for details.