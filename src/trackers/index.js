const ColorTracker = require('./ColorTracker');
const ObjectTracker = require('./ObjectTracker');
const Tracker = require('./Tracker');
const TrackerTask = require('./TrackerTask');
const LandmarksTracker = require('./LandmarksTracker')

const trackers = {
  ColorTracker,
  LandmarksTracker,
  ObjectTracker,
  Tracker,
  TrackerTask
};

module.exports = trackers;
