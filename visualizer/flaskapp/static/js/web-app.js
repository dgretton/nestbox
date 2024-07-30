import { initVisualization } from '/ui-core/js/app.js';

// Web-specific initialization
const socket = io.connect('http://' + document.location.hostname + ':' + location.port);

// Initialize visualization
initVisualization(socket);

// ... (other web-specific code)