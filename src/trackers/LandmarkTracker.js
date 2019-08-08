const Tracker = require('./Tracker');
const ViolaJones = require('../training').ViolaJones;

/**
 * LandmarkTracker utility.
 * @constructor
 * @param {string|Array.<string|Array.<number>>} opt_classifiers Optional
 *     object classifiers to track.
 * @extends {Tracker}
 */
class LandmarkTracker extends Tracker {
}

/**
 * Tracks the `Video` frames. This method is called for each video frame in
 * order to emit `track` event.
 * @param {Uint8ClampedArray} pixels The pixels data to track.
 * @param {number} width The pixels canvas width.
 * @param {number} height The pixels canvas height.
 */
LandmarkTracker.prototype.track = function(pixels, width, height) {
  const image = {
    'data': pixels,
    'width': width,
    'height': height
  };

  const classifier = ViolaJones.classifiers['face'];

  const faces = ViolaJones.detect(pixels, width, height,
    this.getInitialScale(), this.getScaleFactor(), this.getStepSize(),
    this.getEdgesDensity(), classifier);

  const landmarks = tracking.LBF.align(pixels, width, height, faces);

  this.emit('track', {
    'data': {
      'faces': faces,
      'landmarks': landmarks
    }
  });
};

module.exports = LandmarkTracker;
