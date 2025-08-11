const path = require('path');
const { getDefaultConfig } = require('@react-native/metro-config');
const {
  wrapWithAudioAPIMetroConfig,
} = require('react-native-audio-api/metro-config');
const { getConfig } = require('react-native-builder-bob/metro-config');
const pkg = require('../package.json');

const root = path.resolve(__dirname, '..');

/**
 * Metro configuration
 * https://facebook.github.io/metro/docs/configuration
 *
 * @type {import('metro-config').MetroConfig}
 */
const base = getConfig(getDefaultConfig(__dirname), {
  root,
  pkg,
  project: __dirname,
});

module.exports = wrapWithAudioAPIMetroConfig(base);
