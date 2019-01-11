var express = require('express'),
    router = express.Router(),
    db = require('../models');

router.get('/', (req, res) => {
  db.Images.find({})
  .then((data) => {
      res.send(data);
  })
  .catch((err) => {
      res.send(err);
  });
});

module.exports = router;
