const path = require('path');
const CopyPlugin = require("copy-webpack-plugin");
module.exports = {
  /**
   * This is the main entry point for your application, it's the first file
   * that runs in the main process.
   */
  entry: './src/main.js',
  // Put your normal webpack config below here
  module: {
    rules: require('./webpack.rules'),
  },
  plugins: [
    new CopyPlugin({
      patterns: [
        { from: 'src/index.html', to: 'src/index.html' },
        { from: '../../ui-core', to: 'ui-core' },
        { from: '../../libs', to: 'libs' },
        { from: 'nestbox-tray-icon.png', to: 'nestbox-tray-icon.png' }
      ],
    }),
  ],
  node: {
    __dirname: false,
    __filename: false
  }
};
