const ColorTracker = require('./ColorTracker');
const ObjectTracker = require('./ObjectTracker');
const Tracker = require('./Tracker');
const TrackerTask = require('./TrackerTask');
const LandmarkTracker = require('./LandmarkTracker')

const trackers = {
  ColorTracker,
  LandmarkTracker,
  ObjectTracker,
  Tracker,
  TrackerTask
};

module.exports = trackers;
