# Nestbox UI and Visualization

This directory contains the desktop application for Nestbox config, monitoring and visualization. The visualization and UI components are structured to be shared between a web-based version and an Electron-based desktop application.

## Project Structure

- `ui-core/`: Contains shared UI components and logic used by both web and desktop versions.
  - `css/`: Stylesheets for the UI.
  - `js/`: JavaScript modules for 3D visualization and UI logic.
- `desktop/`: Electron-specific files for the desktop application.
- `visualizer/`: Web-specific files and Flask application.
  - `flaskapp/`: Flask application for serving the web version.
    - `static/`: Static files for the web app.
    - `templates/`: HTML templates for the web app.

## Setup

1. Ensure you have Python installed. 3.7 has been used for development but newer versions should work.
2. Optionally, create a virtual environment:
```bash
cd ./python
python -m venv venv
# or
virtualenv venv
source venv/bin/activate
# on windows:
venv\Scripts\activate
```
3. Install the required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
4. If you want to test-run the aligner process to see visuals in the app, you will need to install redis:
```bash
brew install redis # on mac
sudo apt-get install redis-server # on linux
```
On windows you can download the installer from the redis website: https://redis.io/download

## Running the Flask App

To run the web-based visualization:

1. Ensure you have Python and Flask installed (see Setup above, confirm with `pip freeze`).
2. Navigate to the `/visualizer/flaskapp` directory.
3. Run the Flask app:
```bash
python app.py
```
4. Open a web browser and go to `http://localhost:5000`.

## Running the aligner

To run the aligner:

1. Ensure Redis is installed (see Setup above).
2. Make sure Redis is running on your machine. In a separate terminal run the following command:
```bash
redis-server
```
3. While the Flask app is running, open another new terminal and run the following command:
```bash
python nestbox/test_aligner.py
```

## Integrating UI-Core into Electron App

To start integrating the shared UI components into the Electron app:

1. Ensure you have Node.js and npm installed.
2. Navigate to the `desktop/` directory.
3. Install Electron and other dependencies:
```bash
npm install electron
```
4. Update `main.js` to load `index.html`:
```javascript
mainWindow.loadFile('index.html');
```
5. Edit index.html in the desktop/ directory, referencing the shared UI files
6. Create renderer.js in the desktop/ directory to handle Electron-specific logic:
```javascript
import { initVisualization } from '../ui-core/js/app.js';

// Initialize visualization
initVisualization();

// Add Electron-specific code here
```
7. Update package.json to include the main script: (example from the Electron Quick Start)
```json
{
  "name": "my-electron-app",
  "version": "1.0.0",
  "description": "Hello World!",
  "main": "main.js",
  "scripts": {
    "start": "electron ."
  },
  "author": "Jane Doe",
  "license": "MIT",
  "devDependencies": {
    "electron": "23.1.3"
  }
}
```
run the Electron app:
```bash
npm start
```

## Development Workflow

- Make changes to shared components in ui-core/.
- Test changes in the web version by running the Flask app.
- Integrate and test changes in the Electron app.
- Repeat as necessary, keeping the shared code in sync between web and desktop versions.

## Notes

- Ensure that file paths in your HTML and JavaScript files correctly reference the shared ui-core directory.
- When developing, you may need to refresh the Electron app to see changes (Ctrl+R or Cmd+R).
- For production, you'll need to set up a build process to package the Electron app with all necessary dependencies.
