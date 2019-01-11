var mongoose = require('mongoose');

mongoose.set('debug', true);
mongoose.connect('mongodb://localhost:27017/data', { useNewUrlParser: true });
mongoose.Promise = Promise;

module.exports.Images = require('./Images');
