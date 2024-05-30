import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
//import io from 'socket.io';
import { CoordinateSystem, GridEnvironment, PointCollection, Line, Cube, UncertaintyEllipsoid} from './nestboxvis.js';

var scene, camera, renderer, controls;
var coordinateSystems;
var visualElements;
var gridEnvironment; // Instance of GridEnvironment

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    // Background
    renderer.setClearColor(0x222222, 1);
    document.body.appendChild(renderer.domElement);

    // Initialize controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;

    // Create environment
    gridEnvironment = new GridEnvironment(scene);

    // Create coordinate origin map from names to CoordinateSystem instances
    coordinateSystems = {};

    // Create visual element map from names to visual element instances
    visualElements = {};

    camera.position.z = 5;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

init();
animate();

// Connect to WebSocket server
const socket = io.connect('http://' + document.location.hostname + ':' + location.port);

// Menu actions
function toggleMenu() {
    const settingsPanel = document.getElementById('settingsPanel');
    settingsPanel.classList.toggle('hidden');
}

window.toggleMenu = toggleMenu;

function updateCoordinateSystemList() {
    const select = document.getElementById('pinSystem');
    // Clear existing options
    select.innerHTML = '<option value="none">None</option>';
    Object.keys(coordinateSystems).forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
    });
}

function applyPin() {
    const selectedSystem = document.getElementById('pinSystem').value;
    // send just the index of the pinned system
    socket.emit('pin_system', { pin: selectedSystem });
}

window.applyPin = applyPin;

// Listen for optimization updates
socket.on('optimization_update', function(data) {
    console.log('Received optimization update:', data);
    // data has an attribute 'coordinateSystems' that contains a list of coordinate system data structures
    // first we'll create some new coordinate systems based on the data if they don't already exist in the scene
    data.coordinateSystems.forEach(coordSystemData => {
        const name = coordSystemData.name;
        if (!(name in coordinateSystems)) {
            coordinateSystems[name] = new CoordinateSystem(scene);
            updateCoordinateSystemList();
        }
        coordinateSystems[name].update(coordSystemData);
    });
    data.visualElements.forEach(visualElementData => {
        const parent = visualElementData.parent;
        var parentObject;
        if (parent in coordinateSystems) {
            parentObject = coordinateSystems[parent].object3d;
        } else {
            parentObject = scene;
        }
        const type = visualElementData.type;
        const name = visualElementData.name;
        if (type === 'point_collection') {
            if (!(name in visualElements)) {
                visualElements[name] = new PointCollection(parentObject);
            }
        } else if (type === 'line') {
            if (!(name in visualElements)) {
                visualElements[name] = new Line(parentObject);
            }
        } else if (type === 'cube') {
            if (!(name in visualElements)) {
                visualElements[name] = new Cube(parentObject);
            }
        } else if (type === 'ellipsoid') {
            if (!(name in visualElements)) {
                visualElements[name] = new UncertaintyEllipsoid(parentObject);
            }
        }
        else {
            console.error('Unknown visual element type: ', type);
            return;
        }
        visualElements[name].update(visualElementData.properties);
    });
});