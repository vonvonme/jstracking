let LBF = {};

/**
 * LBF Regressor utility.
 * @constructor
 */
LBF.Regressor = function (maxNumStages) {
  this.maxNumStages = maxNumStages;

  this.rfs = new Array(maxNumStages);
  this.models = new Array(maxNumStages);
  for (var i = 0; i < maxNumStages; i++) {
    this.rfs[i] = new LBF.RandomForest(i);
    this.models[i] = LBF.RegressorData[i].models;
  }

  this.meanShape = LBF.LandmarksData;
}

/**
 * Predicts the position of the landmarks based on the bounding box of the face.
 * @param {pixels} pixels The grayscale pixels in a linear array.
 * @param {number} width Width of the image.
 * @param {number} height Height of the image.
 * @param {object} boudingBox Bounding box of the face to be aligned.
 * @return {matrix} A matrix with each landmark position in a row [x,y].
 */
LBF.Regressor.prototype.predict = function (pixels, width, height, boundingBox) {

  var images = [];
  var currentShapes = [];
  var boundingBoxes = [];

  var meanShapeClone = tracking.Matrix.clone(this.meanShape);

  images.push({
    'data': pixels,
    'width': width,
    'height': height
  });
  boundingBoxes.push(boundingBox);

  currentShapes.push(LBF.projectShapeToBoundingBox_(meanShapeClone, boundingBox));

  for (var stage = 0; stage < this.maxNumStages; stage++) {
    var binaryFeatures = LBF.Regressor.deriveBinaryFeat(this.rfs[stage], images, currentShapes, boundingBoxes, meanShapeClone);
    this.applyGlobalPrediction(binaryFeatures, this.models[stage], currentShapes, boundingBoxes);
  }

  return currentShapes[0];
};

/**
 * Multiplies the binary features of the landmarks with the regression matrix
 * to obtain the displacement for each landmark. Then applies this displacement
 * into the landmarks shape.
 * @param {object} binaryFeatures The binary features for the landmarks.
 * @param {object} models The regressor models.
 * @param {matrix} currentShapes The landmarks shapes.
 * @param {array} boudingBoxes The bounding boxes of the faces.
 */
LBF.Regressor.prototype.applyGlobalPrediction = function (binaryFeatures, models, currentShapes,
                                                          boundingBoxes) {

  var residual = currentShapes[0].length * 2;

  var rotation = [];
  var deltashape = new Array(residual / 2);
  for (var i = 0; i < residual / 2; i++) {
    deltashape[i] = [0.0, 0.0];
  }

  for (var i = 0; i < currentShapes.length; i++) {
    for (var j = 0; j < residual; j++) {
      var tmp = 0;
      for (var lx = 0, idx = 0; (idx = binaryFeatures[i][lx].index) != -1; lx++) {
        if (idx <= models[j].nr_feature) {
          tmp += models[j].data[(idx - 1)] * binaryFeatures[i][lx].value;
        }
      }
      if (j < residual / 2) {
        deltashape[j][0] = tmp;
      } else {
        deltashape[j - residual / 2][1] = tmp;
      }
    }

    var res = LBF.similarityTransform_(LBF.unprojectShapeToBoundingBox_(currentShapes[i], boundingBoxes[i]), this.meanShape);
    var rotation = tracking.Matrix.transpose(res[0]);

    var s = LBF.unprojectShapeToBoundingBox_(currentShapes[i], boundingBoxes[i]);
    s = tracking.Matrix.add(s, deltashape);

    currentShapes[i] = LBF.projectShapeToBoundingBox_(s, boundingBoxes[i]);

  }
};

/**
 * Derives the binary features from the image for each landmark.
 * @param {object} forest The random forest to search for the best binary feature match.
 * @param {array} images The images with pixels in a grayscale linear array.
 * @param {array} currentShapes The current landmarks shape.
 * @param {array} boudingBoxes The bounding boxes of the faces.
 * @param {matrix} meanShape The mean shape of the current landmarks set.
 * @return {array} The binary features extracted from the image and matched with the
 *     training data.
 * @static
 */
LBF.Regressor.deriveBinaryFeat = function (forest, images, currentShapes, boundingBoxes, meanShape) {

  var binaryFeatures = new Array(images.length);
  for (var i = 0; i < images.length; i++) {
    var t = forest.maxNumTrees * forest.landmarkNum + 1;
    binaryFeatures[i] = new Array(t);
    for (var j = 0; j < t; j++) {
      binaryFeatures[i][j] = {};
    }
  }

  var leafnodesPerTree = 1 << (forest.maxDepth - 1);

  for (var i = 0; i < images.length; i++) {

    var projectedShape = LBF.unprojectShapeToBoundingBox_(currentShapes[i], boundingBoxes[i]);
    var transform = LBF.similarityTransform_(projectedShape, meanShape);

    for (var j = 0; j < forest.landmarkNum; j++) {
      for (var k = 0; k < forest.maxNumTrees; k++) {

        var binaryCode = LBF.Regressor.getCodeFromTree(forest.rfs[j][k], images[i],
          currentShapes[i], boundingBoxes[i], transform[0], transform[1]);

        var index = j * forest.maxNumTrees + k;
        binaryFeatures[i][index].index = leafnodesPerTree * index + binaryCode;
        binaryFeatures[i][index].value = 1;

      }
    }
    binaryFeatures[i][forest.landmarkNum * forest.maxNumTrees].index = -1;
    binaryFeatures[i][forest.landmarkNum * forest.maxNumTrees].value = -1;
  }
  return binaryFeatures;

}

/**
 * Gets the binary code for a specific tree in a random forest. For each landmark,
 * the position from two pre-defined points are recovered from the training data
 * and then the intensity of the pixels corresponding to these points is extracted
 * from the image and used to traverse the trees in the random forest. At the end,
 * the ending nodes will be represented by 1, and the remaining nodes by 0.
 *
 * +--------------------------- Random Forest -----------------------------+
 * | Ø = Ending leaf                                                       |
 * |                                                                       |
 * |       O             O             O             O             O       |
 * |     /   \         /   \         /   \         /   \         /   \     |
 * |    O     O       O     O       O     O       O     O       O     O    |
 * |   / \   / \     / \   / \     / \   / \     / \   / \     / \   / \   |
 * |  Ø   O O   O   O   O Ø   O   O   Ø O   O   O   O Ø   O   O   O O   Ø  |
 * |  1   0 0   0   0   0 1   0   0   1 0   0   0   0 1   0   0   0 0   1  |
 * +-----------------------------------------------------------------------+
 * Final binary code for this landmark: 10000010010000100001
 *
 * @param {object} forest The tree to be analyzed.
 * @param {array} image The image with pixels in a grayscale linear array.
 * @param {matrix} shape The current landmarks shape.
 * @param {object} boudingBoxes The bounding box of the face.
 * @param {matrix} rotation The rotation matrix used to transform the projected landmarks
 *     into the mean shape.
 * @param {number} scale The scale factor used to transform the projected landmarks
 *     into the mean shape.
 * @return {number} The binary code extracted from the tree.
 * @static
 */
LBF.Regressor.getCodeFromTree = function (tree, image, shape, boundingBox, rotation, scale) {
  var current = 0;
  var bincode = 0;

  while (true) {

    var x1 = Math.cos(tree.nodes[current].feats[0]) * tree.nodes[current].feats[2] * tree.maxRadioRadius * boundingBox.width;
    var y1 = Math.sin(tree.nodes[current].feats[0]) * tree.nodes[current].feats[2] * tree.maxRadioRadius * boundingBox.height;
    var x2 = Math.cos(tree.nodes[current].feats[1]) * tree.nodes[current].feats[3] * tree.maxRadioRadius * boundingBox.width;
    var y2 = Math.sin(tree.nodes[current].feats[1]) * tree.nodes[current].feats[3] * tree.maxRadioRadius * boundingBox.height;

    var project_x1 = rotation[0][0] * x1 + rotation[0][1] * y1;
    var project_y1 = rotation[1][0] * x1 + rotation[1][1] * y1;

    var real_x1 = Math.floor(project_x1 + shape[tree.landmarkID][0]);
    var real_y1 = Math.floor(project_y1 + shape[tree.landmarkID][1]);
    real_x1 = Math.max(0.0, Math.min(real_x1, image.height - 1.0));
    real_y1 = Math.max(0.0, Math.min(real_y1, image.width - 1.0));

    var project_x2 = rotation[0][0] * x2 + rotation[0][1] * y2;
    var project_y2 = rotation[1][0] * x2 + rotation[1][1] * y2;

    var real_x2 = Math.floor(project_x2 + shape[tree.landmarkID][0]);
    var real_y2 = Math.floor(project_y2 + shape[tree.landmarkID][1]);
    real_x2 = Math.max(0.0, Math.min(real_x2, image.height - 1.0));
    real_y2 = Math.max(0.0, Math.min(real_y2, image.width - 1.0));
    var pdf = Math.floor(image.data[real_y1 * image.width + real_x1]) -
      Math.floor(image.data[real_y2 * image.width + real_x2]);

    if (pdf < tree.nodes[current].thresh) {
      current = tree.nodes[current].cnodes[0];
    } else {
      current = tree.nodes[current].cnodes[1];
    }

    if (tree.nodes[current].is_leafnode == 1) {
      bincode = 1;
      for (var i = 0; i < tree.leafnodes.length; i++) {
        if (tree.leafnodes[i] == current) {
          return bincode;
        }
        bincode++;
      }
      return bincode;
    }

  }

  return bincode;
}

/**
 * Face Alignment via Regressing Local Binary Features (LBF)
 * This approach has two components: a set of local binary features and
 * a locality principle for learning those features.
 * The locality principle is used to guide the learning of a set of highly
 * discriminative local binary features for each landmark independently.
 * The obtained local binary features are used to learn a linear regression
 * that later will be used to guide the landmarks in the alignment phase.
 *
 * @authors: VoxarLabs Team (http://cin.ufpe.br/~voxarlabs)
 *           Lucas Figueiredo <lsf@cin.ufpe.br>, Thiago Menezes <tmc2@cin.ufpe.br>,
 *           Thiago Domingues <tald@cin.ufpe.br>, Rafael Roberto <rar3@cin.ufpe.br>,
 *           Thulio Araujo <tlsa@cin.ufpe.br>, Joao Victor <jvfl@cin.ufpe.br>,
 *           Tomer Simis <tls@cin.ufpe.br>)
 */

/**
 * Holds the maximum number of stages that will be used in the alignment algorithm.
 * Each stage contains a different set of random forests and retrieves the binary
 * code from a more "specialized" (i.e. smaller) region around the landmarks.
 * @type {number}
 * @static
 */
LBF.maxNumStages = 4;

/**
 * Holds the regressor that will be responsible for extracting the local features from
 * the image and guide the landmarks using the training data.
 * @type {object}
 * @protected
 * @static
 */
LBF.regressor_ = null;

/**
 * Generates a set of landmarks for a set of faces
 * @param {pixels} pixels The pixels in a linear [r,g,b,a,...] array.
 * @param {number} width The image width.
 * @param {number} height The image height.
 * @param {array} faces The list of faces detected in the image
 * @return {array} The aligned landmarks, each set of landmarks corresponding
 *     to a specific face.
 * @static
 */
LBF.align = function (pixels, width, height, faces) {

  if (LBF.regressor_ == null) {
    LBF.regressor_ = new LBF.Regressor(
      LBF.maxNumStages
    );
  }
// NOTE: is this thesholding suitable ? if it is on image, why no skin-color filter ? and a adaptative threshold
  pixels = tracking.Image.grayscale(pixels, width, height, false);

  pixels = tracking.Image.equalizeHist(pixels, width, height);

  var shapes = new Array(faces.length);

  for (var i in faces) {

    faces[i].height = faces[i].width;

    var boundingBox = {};
    boundingBox.startX = faces[i].x;
    boundingBox.startY = faces[i].y;
    boundingBox.width = faces[i].width;
    boundingBox.height = faces[i].height;

    shapes[i] = LBF.regressor_.predict(pixels, width, height, boundingBox);
  }

  return shapes;
}

/**
 * Unprojects the landmarks shape from the bounding box.
 * @param {matrix} shape The landmarks shape.
 * @param {matrix} boudingBox The bounding box.
 * @return {matrix} The landmarks shape projected into the bounding box.
 * @static
 * @protected
 */
LBF.unprojectShapeToBoundingBox_ = function (shape, boundingBox) {
  var temp = new Array(shape.length);
  for (var i = 0; i < shape.length; i++) {
    temp[i] = [
      (shape[i][0] - boundingBox.startX) / boundingBox.width,
      (shape[i][1] - boundingBox.startY) / boundingBox.height
    ];
  }
  return temp;
}

/**
 * Projects the landmarks shape into the bounding box. The landmarks shape has
 * normalized coordinates, so it is necessary to map these coordinates into
 * the bounding box coordinates.
 * @param {matrix} shape The landmarks shape.
 * @param {matrix} boudingBox The bounding box.
 * @return {matrix} The landmarks shape.
 * @static
 * @protected
 */
LBF.projectShapeToBoundingBox_ = function (shape, boundingBox) {
  var temp = new Array(shape.length);
  for (var i = 0; i < shape.length; i++) {
    temp[i] = [
      shape[i][0] * boundingBox.width + boundingBox.startX,
      shape[i][1] * boundingBox.height + boundingBox.startY
    ];
  }
  return temp;
}

/**
 * Calculates the rotation and scale necessary to transform shape1 into shape2.
 * @param {matrix} shape1 The shape to be transformed.
 * @param {matrix} shape2 The shape to be transformed in.
 * @return {[matrix, scalar]} The rotation matrix and scale that applied to shape1
 *     results in shape2.
 * @static
 * @protected
 */
LBF.similarityTransform_ = function (shape1, shape2) {

  var center1 = [0, 0];
  var center2 = [0, 0];
  for (var i = 0; i < shape1.length; i++) {
    center1[0] += shape1[i][0];
    center1[1] += shape1[i][1];
    center2[0] += shape2[i][0];
    center2[1] += shape2[i][1];
  }
  center1[0] /= shape1.length;
  center1[1] /= shape1.length;
  center2[0] /= shape2.length;
  center2[1] /= shape2.length;

  var temp1 = tracking.Matrix.clone(shape1);
  var temp2 = tracking.Matrix.clone(shape2);
  for (var i = 0; i < shape1.length; i++) {
    temp1[i][0] -= center1[0];
    temp1[i][1] -= center1[1];
    temp2[i][0] -= center2[0];
    temp2[i][1] -= center2[1];
  }

  var covariance1, covariance2;
  var mean1, mean2;

  var t = tracking.Matrix.calcCovarMatrix(temp1);
  covariance1 = t[0];
  mean1 = t[1];

  t = tracking.Matrix.calcCovarMatrix(temp2);
  covariance2 = t[0];
  mean2 = t[1];

  var s1 = Math.sqrt(tracking.Matrix.norm(covariance1));
  var s2 = Math.sqrt(tracking.Matrix.norm(covariance2));

  var scale = s1 / s2;
  temp1 = tracking.Matrix.mulScalar(1.0 / s1, temp1);
  temp2 = tracking.Matrix.mulScalar(1.0 / s2, temp2);

  var num = 0, den = 0;
  for (var i = 0; i < shape1.length; i++) {
    num = num + temp1[i][1] * temp2[i][0] - temp1[i][0] * temp2[i][1];
    den = den + temp1[i][0] * temp2[i][0] + temp1[i][1] * temp2[i][1];
  }

  var norm = Math.sqrt(num * num + den * den);
  var sin_theta = num / norm;
  var cos_theta = den / norm;
  var rotation = [
    [cos_theta, -sin_theta],
    [sin_theta, cos_theta]
  ];

  return [rotation, scale];
}

/**
 * LBF Random Forest data structure.
 * @static
 * @constructor
 */
LBF.RandomForest = function (forestIndex) {
  this.maxNumTrees = LBF.RegressorData[forestIndex].max_numtrees;
  this.landmarkNum = LBF.RegressorData[forestIndex].num_landmark;
  this.maxDepth = LBF.RegressorData[forestIndex].max_depth;
  this.stages = LBF.RegressorData[forestIndex].stages;

  this.rfs = new Array(this.landmarkNum);
  for (var i = 0; i < this.landmarkNum; i++) {
    this.rfs[i] = new Array(this.maxNumTrees);
    for (var j = 0; j < this.maxNumTrees; j++) {
      this.rfs[i][j] = new LBF.Tree(forestIndex, i, j);
    }
  }
}

/**
 * LBF Tree data structure.
 * @static
 * @constructor
 */
LBF.Tree = function (forestIndex, landmarkIndex, treeIndex) {
  var data = LBF.RegressorData[forestIndex].landmarks[landmarkIndex][treeIndex];
  this.maxDepth = data.max_depth;
  this.maxNumNodes = data.max_numnodes;
  this.nodes = data.nodes;
  this.landmarkID = data.landmark_id;
  this.numLeafnodes = data.num_leafnodes;
  this.numNodes = data.num_nodes;
  this.maxNumFeats = data.max_numfeats;
  this.maxRadioRadius = data.max_radio_radius;
  this.leafnodes = data.id_leafnodes;
}

LBF.RegressorData = require('./training/Regressor')
LBF.LandmarksData = require('./training/Landmarks')

module.exports = LBF;
