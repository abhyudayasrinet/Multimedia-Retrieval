var mongoose = require('mongoose');

var Schema = new mongoose.Schema({
  images: Array,
});

var Images = mongoose.model('Images', Schema);

module.exports = Images;
