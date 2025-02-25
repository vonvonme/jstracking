const ObjectTracker = require('./ObjectTracker');
const ViolaJones = require('../training').ViolaJones;

/**
 * LandmarksTracker utility.
 * @constructor
 * @param {string|Array.<string|Array.<number>>} opt_classifiers Optional
 *     object classifiers to track.
 * @extends {ObjectTracker}
 */
class LandmarksTracker extends ObjectTracker {
  constructor(opt_classifiers) {
    super(opt_classifiers);
  }
}

/**
 * Tracks the `Video` frames. This method is called for each video frame in
 * order to emit `track` event.
 * @param {Uint8ClampedArray} pixels The pixels data to track.
 * @param {number} width The pixels canvas width.
 * @param {number} height The pixels canvas height.
 */
LandmarksTracker.prototype.track = function(pixels, width, height) {
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

module.exports = LandmarksTracker;
