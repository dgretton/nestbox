// main.js

const { app, BrowserWindow, Tray, Menu } = require('electron');
const { spawn } = require('child_process');
const path = require('path');
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const redis = require('ioredis');

const expressApp = express();
const server = http.createServer(expressApp);
const port = 5000;

let redisServer;

let mainWindow;

// Electron setup
const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    resizable: true,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    }
  });

  // For debugging
  mainWindow.webContents.openDevTools()

  mainWindow.loadURL(`http://localhost:${port}`);

  mainWindow.on('minimize', function (event) {
    event.preventDefault();
    mainWindow.hide();
  });

  mainWindow.on('close', function (event) {
    if (!app.isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
    return false;
  });
};

const createTray = () => {
  const tray = new Tray(path.join(__dirname, 'nestbox-tray-icon.png'));
  const contextMenu = Menu.buildFromTemplate([
    { label: 'Show App', click: () => mainWindow.show() },
    { label: 'Quit', click: () => { app.isQuitting = true; app.quit(); } }
  ]);
  tray.setToolTip('Nestbox App');
  tray.setContextMenu(contextMenu);
}

app.whenReady().then(() => {
  startRedisServer();
  setupServer();
  createWindow();
  createTray();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Quit the app when closed
app.on('will-quit', () => {
  server.close();
  redisServer.kill();
});

function startRedisServer() {
  redisServer = spawn('redis-server');

  redisServer.stdout.on('data', (data) => {
      console.log(`Redis Server: ${data}`);
  });

  redisServer.stderr.on('data', (data) => {
      console.error(`Redis Server Error: ${data}`);
  });

  redisServer.on('close', (code) => {
      console.log(`Redis Server process exited with code ${code}`);
  });
}

// Express server setup
function setupServer() {
  const io = socketIo(server, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"]
    }
  });
  const redisClient = redis.createClient();
  redisClient.connect();

  // Serve static files
  expressApp.use('/ui-core', express.static(path.join(__dirname, 'ui-core')));
  expressApp.use('/libs', express.static(path.join(__dirname, 'libs')));

  // Serve index.html
  expressApp.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, "src", "index.html"));
  });

  // Socket.IO connection
  io.on('connection', (socket) => {
    console.log('A client connected', socket.id);

    socket.on('pin_system', (message) => {
      console.log(`Pin coordinate system: ${message.pin}`);
      redisClient.publish('pin_command', JSON.stringify({ pin: message.pin }));
    });

    socket.on('disconnect', () => {
      console.log('A client disconnected');
    });

    socket.on("test", (data) => {
      console.log(data);
      io.emit("test_back", "testing back");
    });
  });

  // Redis Pub/Sub Listener
  const redisSub = redisClient.duplicate();
  redisSub.subscribe('optimization_update');
  redisSub.on('message', (channel, message) => {
    if (channel === 'optimization_update') {
      io.emit('optimization_update', JSON.parse(message));
    }
  });

  server.listen(port, () => {
    console.log(`Server is running on port ${port}`);
  });
}

// const { app, BrowserWindow } = require('electron');
// const path = require('node:path');

// // Handle creating/removing shortcuts on Windows when installing/uninstalling.
// if (require('electron-squirrel-startup')) {
//   app.quit();
// }

// const createWindow = () => {
//   // Create the browser window.
//   const mainWindow = new BrowserWindow({
//     width: 800,
//     height: 600,
//     webPreferences: {
//       preload: MAIN_WINDOW_PRELOAD_WEBPACK_ENTRY,
//     },
//   });

//   // and load the index.html of the app.
//   mainWindow.loadURL(MAIN_WINDOW_WEBPACK_ENTRY);

//   // Open the DevTools.
//   mainWindow.webContents.openDevTools();
// };

// // This method will be called when Electron has finished
// // initialization and is ready to create browser windows.
// // Some APIs can only be used after this event occurs.
// app.whenReady().then(() => {
//   createWindow();

//   // On OS X it's common to re-create a window in the app when the
//   // dock icon is clicked and there are no other windows open.
//   app.on('activate', () => {
//     if (BrowserWindow.getAllWindows().length === 0) {
//       createWindow();
//     }
//   });
// });

// // Quit when all windows are closed, except on macOS. There, it's common
// // for applications and their menu bar to stay active until the user quits
// // explicitly with Cmd + Q.
// app.on('window-all-closed', () => {
//   if (process.platform !== 'darwin') {
//     app.quit();
//   }
// });

// // In this file you can include the rest of your app's specific main process
// // code. You can also put them in separate files and import them here.
